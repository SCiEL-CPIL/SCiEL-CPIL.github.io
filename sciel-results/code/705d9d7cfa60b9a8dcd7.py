import numpy as np
import time
import random

def construct():
    N = 18
    MIN_DIST = 6
    deadline = time.time() + 80  # 80s safety margin
    
    # Generate all weight 9 words
    # There are C(18, 9) = 48620 such words.
    candidates = []
    # Efficient generation of combinations
    # Or just iterate all 2^18 and filter? 262k is small enough.
    # But generating weight 9 directly is cleaner.
    
    # Let's just iterate all 2^18 and filter by bit_count == 9
    # This ensures we consider a large, symmetric set of "central" codewords.
    # We can also include weight 8 and 10 later if time permits, but 9 is the sweet spot for distance.
    
    all_words = []
    for w in range(1 << N):
        if w.bit_count() == 9:
            all_words.append(w)
            
    # Shuffle to avoid bias
    random.seed(42)
    random.shuffle(all_words)
    
    code = []
    
    # Greedy packing
    for w in all_words:
        if time.time() > deadline:
            break
            
        is_valid = True
        for c in code:
            if (w ^ c).bit_count() < MIN_DIST:
                is_valid = False
                break
                
        if is_valid:
            code.append(w)
            
    # If we have time, try to add more from other weights?
    # Or just return what we have.
    # Greedy on weight 9 usually yields a good code.
    # A_2(18,6) is 128. Greedy might find 60-100.
    
    # To improve, we can try a second pass with a different shuffle or include weight 8/10.
    # But let's stick to a robust single pass first.
    
    # If the list is empty (shouldn't happen), return a default
    if not code:
         # Fallback to RM code
         words = []
         for slope in range(16):
             for intercept in range(2):
                 word = 0
                 for point in range(16):
                     bit = ((slope & point).bit_count() + intercept) % 2
                     word |= bit << point
                 words.append(word)
         return words
         
    return code
