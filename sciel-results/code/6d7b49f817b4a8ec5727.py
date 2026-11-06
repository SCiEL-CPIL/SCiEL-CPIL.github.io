import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    N = 17
    MIN_DIST = 6
    
    # Generate the 32-word Reed-Muller core
    def generate_core():
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words
    
    codewords = generate_core()
    codeword_set = set(codewords)
    
    def is_valid(candidate, existing_words):
        for w in existing_words:
            if (candidate ^ w).bit_count() < MIN_DIST:
                return False
        return True
    
    # Greedy pass 1: iterate through all candidates in natural order
    for candidate in range(1, 1 << N):
        if time.time() > deadline:
            break
        if candidate in codeword_set:
            continue
        if is_valid(candidate, codewords):
            codewords.append(candidate)
            codeword_set.add(candidate)
    
    # Greedy pass 2: try random orderings to find more
    best_count = len(codewords)
    
    while time.time() < deadline - 5.0:
        # Create a shuffled list of candidates not yet in codewords
        remaining = [c for c in range(1, 1 << N) if c not in codeword_set]
        if not remaining:
            break
        random.shuffle(remaining)
        
        added_any = False
        for candidate in remaining:
            if time.time() > deadline - 5.0:
                break
            if is_valid(candidate, codewords):
                codewords.append(candidate)
                codeword_set.add(candidate)
                added_any = True
        
        if not added_any:
            break
    
    return codewords
