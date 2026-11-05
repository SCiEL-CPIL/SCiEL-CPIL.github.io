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
    
    # Precompute popcount table for fast distance checks
    popcount_table = np.array([i.bit_count() for i in range(2**17)], dtype=np.int8)
    
    # Step 2: Greedy extension with vectorized distance checks
    best_code = current_code
    best_size = len(current_code)
    
    # Try multiple restarts with different random seeds
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        rng = np.random.default_rng(seed=restart)
        
        # Start with the base code
        current_list = list(current_code)
        current_array = np.array(current_list, dtype=np.int64)
        
        # Generate all candidate words (all 17-bit words)
        all_words = np.arange(2**17, dtype=np.int64)
        
        # Shuffle candidates
        candidates = rng.permutation(all_words)
        
        # Process candidates in batches for vectorized distance computation
        batch_size = 10000
        
        for batch_start in range(0, len(candidates), batch_size):
            if time.time() > deadline - 2.0:
                break
                
            batch_end = min(batch_start + batch_size, len(candidates))
            batch = candidates[batch_start:batch_end]
            
            # For each candidate in the batch, compute min distance to current code
            # Vectorize: for each candidate w, compute (current_array ^ w) and popcount
            # We process candidates one at a time but use numpy for the distance to all codewords
            
            for w in batch:
                if time.time() > deadline - 2.0:
                    break
                
                # Compute distances from w to all current codewords
                diffs = current_array ^ w
                dists = popcount_table[diffs]
                min_d = np.min(dists)
                
                if min_d >= 6:
                    current_list.append(int(w))
                    current_array = np.append(current_array, w)
        
        if len(current_list) > best_size:
            best_size = len(current_list)
            best_code = np.array(current_list, dtype=np.int64)
            
    # Return the best code found
    return list(best_code)
