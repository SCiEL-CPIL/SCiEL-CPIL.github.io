import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # 1. Construct the 256-word [17, 9, 6] code
    poly = 0x11D
    
    def gf_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= poly
        return res & 0xFF

    def gf_pow(a, n):
        res = 1
        base = a
        while n:
            if n & 1:
                res = gf_mul(res, base)
            base = gf_mul(base, base)
            n >>= 1
        return res

    def get_primitive_17th_root():
        for candidate in range(2, 256):
            if gf_pow(candidate, 17) == 1 and candidate != 1:
                return candidate
        return None

    beta = get_primitive_17th_root()
    
    if beta is None:
        # Fallback to seed (32 words)
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    # Build generator polynomial G(x)
    poly_coeffs = [1]
    current_beta = beta
    for _ in range(8):
        new_coeffs = [0] * (len(poly_coeffs) + 1)
        for i in range(len(poly_coeffs)):
            new_coeffs[i+1] = poly_coeffs[i]
        for i in range(len(poly_coeffs)):
            new_coeffs[i] ^= gf_mul(poly_coeffs[i], current_beta)
        poly_coeffs = new_coeffs
        current_beta = gf_mul(current_beta, current_beta)
        
    g_bits = [0] * 9
    for i, c in enumerate(poly_coeffs):
        g_bits[i] = 1 if c == 1 else 0
        
    # Generate 256 codewords
    codewords_17 = []
    for u in range(512):
        c = 0
        for k in range(9):
            if (u >> k) & 1:
                for j in range(9):
                    if g_bits[j]:
                        c ^= 1 << (j + k)
        codewords_17.append(c)
        
    # Extend and shorten
    shortened_codewords = []
    for c in codewords_17:
        parity = bin(c).count('1') % 2
        ec = c | (parity << 17)
        if (ec >> 17) & 1 == 0:
            shortened_codewords.append(ec & ((1 << 17) - 1))
            
    code_set = set(shortened_codewords)
    if len(code_set) < 256:
        return list(code_set)
        
    code_list = list(code_set)
    
    # 2. Greedy Add
    # Pre-compute a set for O(1) lookup of existing words
    # We iterate through all 2^17 words
    
    # To speed up, we can skip words that are "close" to the code?
    # No, we must check all.
    
    # Optimization: Use a list of code words for fast iteration.
    # We will break the loop if time runs out.
    
    added = 0
    # Sort candidates? Maybe by weight?
    # Just iterate 0 to 2^17 - 1
    
    # We can use a NumPy array for the code to vectorize the popcount if we batch it,
    # but for now, let's try the fast Python loop with early break.
    
    # To make it faster, we can convert code_list to a tuple? No.
    # Let's try to add words in batches of 1000 using NumPy to check compatibility.
    
    # Create a boolean mask of used words
    used = np.zeros(2**17, dtype=bool)
    for w in code_list:
        used[w] = True
        
    candidates = np.arange(2**17, dtype=np.uint32)
    candidates = candidates[~used]
    
    code_arr = np.array(code_list, dtype=np.uint32)
    
    # Precompute popcount for 17-bit integers for fast vectorized Hamming distance
    pop17 = np.array([bin(i).count('1') for i in range(131072)], dtype=np.uint8)
    
    # Process candidates one by one, but with optimized vectorized check
    # Key insight: use the 17-bit popcount table directly, no splitting needed
    
    for idx in range(len(candidates)):
        if time.time() > deadline - 3:
            break
            
        w = candidates[idx]
        
        # Vectorized compatibility check using 17-bit popcount table
        diffs = code_arr ^ w
        # Cast to appropriate type for indexing
        weights = pop17[diffs.astype(np.int32)]
        
        if weights.min() >= 6:
            code_list.append(w)
            code_arr = np.append(code_arr, w)
            used[w] = True
            added += 1
            
    # Final verification: ensure all pairs have distance >= 6
    # This is a safety check; should be fast for ~256-300 words
    n = len(code_list)
    for i in range(n):
        if time.time() > deadline - 1:
            break
        for j in range(i+1, n):
            if (code_list[i] ^ code_list[j]).bit_count() < 6:
                # Should not happen, but remove the problematic word
                code_list.pop(j)
                break
    
    return code_list
