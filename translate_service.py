import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer

class TranslateRequest(BaseModel):
    zh: str

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_name = "Helsinki-NLP/opus-mt-zh-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    app.state.tokenizer = tokenizer
    app.state.model = model
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/zhToEn")
async def zhToEn(request: TranslateRequest):
    inputs = app.state.tokenizer([request.zh], return_tensors="pt", padding=True)
    translated = app.state.model.generate(**inputs)
    translated_text = app.state.tokenizer.batch_decode(translated, skip_special_tokens=True)
    return { "en": translated_text[0] }