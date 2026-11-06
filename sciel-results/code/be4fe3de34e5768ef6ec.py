import numpy as np
import random
import time

def construct():
    N = 17
    MIN_DIST = 6
    TOTAL = 2 ** N
    
    # Precompute popcount for all possible 17-bit numbers
    # Using a lookup table or bit operations
    # For 2^17, a simple loop is fast enough (~0.1s)
    popcount = np.zeros(TOTAL, dtype=np.uint8)
    for i in range(TOTAL):
        popcount[i] = bin(i).count('1')
        
    words = np.arange(TOTAL, dtype=np.uint32)
    
    best_codewords = []
    best_size = 0
    
    # Number of restarts
    # We have 60s. Each run takes ~0.1-0.5s. 
    # Let's do 20 restarts to be safe.
    num_restarts = 20
    
    for run in range(num_restarts):
        valid = np.ones(TOTAL, dtype=bool)
        codewords = []
        
        # Optional: Force first word to 0 for the first run to anchor the structure
        if run == 0:
            w = 0
            codewords.append(w)
            xor = np.bitwise_xor(words, w)
            dist = popcount[xor]
            valid &= (dist >= MIN_DIST)
            # Ensure 0 is not picked again (it's dist 0 to itself, so valid[0] is False now)
        else:
            # Random start
            # Pick a random word
            w = random.randint(0, TOTAL - 1)
            codewords.append(w)
            xor = np.bitwise_xor(words, w)
            dist = popcount[xor]
            valid &= (dist >= MIN_DIST)
            
        # Greedy loop
        while True:
            candidates = np.where(valid)[0]
            if len(candidates) == 0:
                break
            
            # Selection strategy
            if run == 0:
                # Lexicographic: pick smallest
                w = candidates[0]
            else:
                # Random
                idx = random.randint(0, len(candidates) - 1)
                w = candidates[idx]
            
            codewords.append(w)
            
            # Update valid mask
            xor = np.bitwise_xor(words, w)
            dist = popcount[xor]
            valid &= (dist >= MIN_DIST)
            
            # Optimization: If we have exceeded the current best, we can keep going
            # But we must finish to know the final size.
            
        if len(codewords) > best_size:
            best_size = len(codewords)
            best_codewords = codewords.copy()
            
        # Early termination if we hit the theoretical upper bound? 
        # Upper bound is 340. Unlikely to hit.
        
    return best_codewords
