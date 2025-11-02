import jieba  # 需要安装: pip install jieba
from bleu import compute_bleu, _get_ngrams
import collections

def tokenize_chinese(text):
    """中文分词"""
    return list(jieba.cut(text))

# 中文BLEU计算示例
def chinese_bleu_example():
    # 原始中文文本
    candidate_text = "我爱自然语言处理"
    reference_texts = ["我喜欢自然语言处理", "我热爱NLP技术"]  # 多个参考译文
    
    # 分词
    candidate_tokens = tokenize_chinese(candidate_text)
    reference_tokens_list = [tokenize_chinese(ref) for ref in reference_texts]
    
    print(f"候选译文: {candidate_tokens}")
    print(f"参考译文: {reference_tokens_list}")
    
    # ✅ 修复：正确的输入格式
    # reference_corpus: 对于每个候选翻译，提供一个参考译文列表
    # translation_corpus: 候选翻译的token列表
    reference_corpus = [reference_tokens_list]  # [[ref1_tokens, ref2_tokens]]
    translation_corpus = [candidate_tokens]     # [candidate_tokens]
    bleu_score = compute_bleu(reference_corpus, translation_corpus, max_order=4, smooth=True)
    
    print(f"\nBLEU分数: {bleu_score[0]:.4f}")
    print(f"n-gram精度: {[f'{p:.4f}' for p in bleu_score[1]]}")
    print(f"翻译长度: {bleu_score[4]}, 参考长度: {bleu_score[5]}")
    print(f"长度比例: {bleu_score[3]:.4f}, 简短惩罚: {bleu_score[2]:.4f}")

# 运行修复后的示例
chinese_bleu_example()
