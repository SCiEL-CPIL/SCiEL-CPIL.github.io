import time
import numpy as np
import itertools

def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 55.0
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET

    # Precompute popcount lookup for 9-bit values
    pop9 = np.zeros(512, dtype=np.uint8)
    for i in range(512):
        pop9[i] = i.bit_count()

    def fast_popcount18(arr):
        """Vectorized popcount for numpy array of 18-bit values"""
        low = (arr & 0x1FF).astype(np.intp)
        high = ((arr >> 9) & 0x1FF).astype(np.intp)
        return pop9[low] + pop9[high]

    words = [0]
    words_set = {0}
    words_arr = np.array(words, dtype=np.uint32)
    
    # Generate candidates by weight
    # Weights 1-5 are invalid because dist(0, w) < 6
    # Start from weight 6
    
    for w in range(6, N + 1):
        if time.time() > deadline - 1.0:
            break
            
        # Generate all 18-bit numbers of weight w
        # Use itertools.combinations
        # This is efficient for small w, but for w=9, C(18,9)=48620
        # Total candidates is 2^18 = 262144
        
        # Optimization: If we have found 512 words, we can stop?
        # The task says "largest possible". 512 is the known max.
        # If we reach 512, we can return immediately.
        if len(words) >= 512:
            return words

        # Iterate through combinations
        # To ensure deterministic order, combinations are generated in lexicographic order
        for comb in itertools.combinations(range(N), w):
            if time.time() > deadline - 0.5:
                break
                
            # Construct the integer
            c = 0
            for pos in comb:
                c |= (1 << pos)
                
            if c in words_set:
                continue
                
            # Check distance to all existing words
            c_val = np.uint32(c)
            xor_vals = np.bitwise_xor(words_arr, c_val).astype(np.uint32)
            bit_counts = fast_popcount18(xor_vals)
            
            if np.min(bit_counts) < MIN_DIST:
                continue
            
            # Add to words
            words.append(c)
            words_set.add(c)
            words_arr = np.append(words_arr, c_val)
            
            if len(words) >= 512:
                return words

    return words
