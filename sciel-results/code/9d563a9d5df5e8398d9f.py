    if best_code is None:
        return []
        
    final_words = [int_to_word(c) for c in best_code]
    return final_words
