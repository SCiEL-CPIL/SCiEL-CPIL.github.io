import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Step 1: Construct the 256-word linear code
    # This is the [17, 8, 6] code derived from the [18, 9, 6] extended QR code.
    
    poly = 0x1D
    
    def gf_mul(a, b):
        res = 0
        for _ in range(8):
            if b & 1:
                res ^= a
            b >>= 1
            if a & 0x80:
                a = (a << 1) ^ poly
            else:
                a <<= 1
            a &= 0xFF
        return res
        
    def gf_pow(a, n):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a)
            a = gf_mul(a, a)
            n >>= 1
        return res
        
    alpha = None
    for a in range(1, 256):
        if gf_pow(a, 17) == 1 and a != 1:
            alpha = a
            break
            
    if alpha is None:
        # Fallback if alpha not found (should not happen)
        return [0]
        
    Q = [1, 2, 4, 8, 9, 13, 15, 16]
    
    g_coeffs = [1]
    for i in Q:
        root = gf_pow(alpha, i)
        new_g = [0] * (len(g_coeffs) + 1)
        for j in range(len(g_coeffs)):
            new_g[j + 1] ^= g_coeffs[j]
            new_g[j] ^= gf_mul(g_coeffs[j], root)
        g_coeffs = new_g
        
    G = np.zeros((9, 17), dtype=np.int64)
    for k in range(9):
        for j in range(9):
            if g_coeffs[j]:
                pos = (j + k) % 17
                G[k, pos] = 1
                
    words_18 = []
    for u in range(512):
        word = np.zeros(17, dtype=np.int64)
        for i in range(9):
            if (u >> i) & 1:
                word ^= G[i]
        parity = int(np.sum(word) % 2)
        word_ext = np.append(word, parity)
        words_18.append(word_ext)
        
    words_short = []
    for w in words_18:
        if w[17] == 0:
            w_short = w[:17]
            val = 0
            for i in range(17):
                if w_short[i]:
                    val |= (1 << i)
            words_short.append(val)
            
    # We expect 256 words here
    if not words_short:
        return [0]
        
    # Convert to numpy array for fast operations
    current_code = np.array(words_short, dtype=np.int64)
    
    # Step 2: Greedy extension
    # We will try to add words from the remaining 2^17 - 256 words.
    # To make it deterministic but varied, we can shuffle the candidates.
    # We'll run a few restarts with different random seeds to find the best code.
    
    best_code = current_code
    best_size = len(current_code)
    
    # Precompute all possible words? No, iterate and check.
    # To speed up, we can check candidates in a specific order.
    # Random order is often good.
    
    num_restarts = 10 # Adjust based on time
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Shuffle the remaining words
        all_words = np.arange(2**17, dtype=np.int64)
        # Mask out the words already in the code
        # Since current_code is small, we can just check membership or use a set
        # For efficiency in the loop, we'll just iterate through all words and skip if in current_code
        # Actually, iterating through 131k words is fast.
        
        # Create a shuffled list of candidates
        # Using np.random.permutation is fast
        rng = np.random.default_rng(seed=restart)
        candidates = rng.permutation(all_words)
        
        # We maintain the current code as a list for easy appending, 
        # but we need fast distance checks.
        # Let's keep it as a numpy array and rebuild if we add many, 
        # or just use a list and convert to array for checks?
        # Checking distance against a list of ~300 ints:
        # dists = [ (c ^ w).bit_count() for c in current_list ]
        # This is ~300 python operations. 131k * 300 = 39M ops. ~1-2 seconds per restart.
        # 10 restarts = 20s. Fits in budget.
        
        current_list = list(current_code)
        
        # Optimization: Precompute bit_count for all possible 17-bit differences?
        # There are 2^17 = 131k possible differences.
        # popcount table:
        popcount_table = np.array([i.bit_count() for i in range(2**17)], dtype=np.int8)
        
        for w in candidates:
            if time.time() > deadline - 2.0:
                break
                
            # Check if w is already in current_list
            # Since we are iterating through all words, and current_list is a subset,
            # we can skip if w is in current_list.
            # A set lookup is O(1).
            # But we don't have a set. Let's maintain a set.
            
            # Actually, simpler: just check distance. If it's 0, it's the same word.
            # But we want to avoid adding duplicates.
            # Since we iterate through unique words, and we only add if dist >= 6,
            # and dist(w, w) = 0 < 6, it won't be added if it's already there?
            # Wait, if w is in current_list, dist(w, w) = 0. The min distance will be 0.
            # So the condition min_dist >= 6 will fail.
            # So we don't need a separate set to prevent duplicates, the distance check handles it.
            
            # Calculate min distance to current_list
            # Vectorized approach:
            # diffs = np.array(current_list) ^ w
            # dists = popcount_table[diffs]
            # min_d = np.min(dists)
            
            # This creates a temporary array of size len(current_list) for every candidate.
            # 131k candidates * 300 size array = 39M elements processed.
            # This might be slow in Python loops if not careful.
            # Let's try to optimize the inner loop.
            
            # Alternative: Iterate through current_list in Python and break early if dist < 6.
            # This is often faster for small lists because of early termination.
            min_d = 17 # max possible
            valid = True
            for c in current_list:
                d = (c ^ w).bit_count()
                if d < 6:
                    valid = False
                    break
                # We could track min_d but we only care if it's < 6
            if valid:
                current_list.append(w)
                
        if len(current_list) > best_size:
            best_size = len(current_list)
            best_code = np.array(current_list, dtype=np.int64)
            
    # Return the best code found
    return list(best_code)
