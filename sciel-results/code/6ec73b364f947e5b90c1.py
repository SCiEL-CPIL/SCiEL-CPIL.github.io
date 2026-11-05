import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # 1. Construct the 256-word [17,8,6] linear code
    # This is the shortened extended quadratic-residue code.
    
    def gf_mul(a, b, poly=0x1d):
        res = 0
        for i in range(8):
            if (b & 1):
                res ^= a
            b >>= 1
            if (a & 0x80):
                a = (a << 1) ^ poly
            else:
                a <<= 1
            a &= 0xFF
        return res

    def gf_pow(a, n, poly=0x1d):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a, poly)
            a = gf_mul(a, a, poly)
            n >>= 1
        return res

    # Find an element of order 17 in GF(2^8)
    alpha = None
    for a in range(1, 256):
        if gf_pow(a, 17) == 1 and a != 1:
            alpha = a
            break
            
    if alpha is None:
        # Fallback if construction fails (should not happen with 0x1d)
        return [0] * 32

    # Quadratic residues mod 17
    Q = [1, 2, 4, 8, 9, 13, 15, 16]
    
    # Construct generator polynomial g(x)
    g_coeffs = [1]
    for i in Q:
        root = gf_pow(alpha, i)
        new_g = [0] * (len(g_coeffs) + 1)
        for j in range(len(g_coeffs)):
            new_g[j+1] ^= g_coeffs[j]
            new_g[j] ^= gf_mul(g_coeffs[j], root)
        g_coeffs = new_g
        
    # Construct Generator Matrix G (9x17)
    G = np.zeros((9, 17), dtype=np.uint8)
    for k in range(9):
        for j in range(9):
            if g_coeffs[j]:
                G[k, (j + k) % 17] = 1
                
    # Generate all 512 codewords of the [17,9,5] code
    words_17 = []
    for u in range(512):
        word = np.zeros(17, dtype=np.uint8)
        for i in range(9):
            if (u >> i) & 1:
                word ^= G[i]
        words_17.append(word)
        
    # Extend to [18,9,6] by adding parity bit
    words_18 = []
    for w in words_17:
        weight = np.sum(w)
        parity = weight % 2
        w_ext = np.append(w, parity)
        words_18.append(w_ext)
        
    # Shorten to [17,8,6] by keeping words with parity 0
    code_words = []
    for w in words_18:
        if w[17] == 0:
            w_short = w[:17]
            val = 0
            for i in range(17):
                if w_short[i]:
                    val |= (1 << i)
            code_words.append(val)
            
    # Convert to NumPy array for vectorized operations
    # Shape: (M, 17), dtype uint8
    M = len(code_words)
    codebook = np.zeros((M, 17), dtype=np.uint8)
    for idx, val in enumerate(code_words):
        for i in range(17):
            if (val >> i) & 1:
                codebook[idx, i] = 1
                
    best_code = code_words[:]
    
    # 2. Greedy search to add more words
    random.seed(42)
    
    # Pre-allocate a buffer for candidate bits
    candidate_bits = np.zeros(17, dtype=np.uint8)
    
    # Helper to check if a candidate (as integer) can be added
    # Using vectorized distance calculation
    def can_add_int(c_int):
        # Convert c_int to bits
        for i in range(17):
            if (c_int >> i) & 1:
                candidate_bits[i] = 1
            else:
                candidate_bits[i] = 0
        
        # Vectorized distance check
        # XOR and sum along axis 1
        dists = np.sum(candidate_bits ^ codebook, axis=1)
        
        # Check if all distances are >= 6
        # np.min is faster than np.all(dists >= 6) for small arrays? 
        # Actually, np.all is fine.
        if len(codebook) == 0:
            return True
            
        # Early exit optimization: if any dist < 6, return False
        # np.any(dists < 6)
        return not np.any(dists < 6)

    # Main search loop
    # We try to add random words.
    # To improve efficiency, we can also try words that are "near" existing ones?
    # But random is often sufficient for the first few additions if they exist.
    # Given the gap 256->340, random might not find many.
    # However, vectorization allows ~100,000+ checks/sec.
    
    while time.time() < deadline - 1.0:
        c = random.randint(0, 2**17 - 1)
        
        # Quick check: if c is already in codebook?
        # We can maintain a set of integers for O(1) lookup
        # But we need to update the set when we add.
        # Let's maintain a set for duplicates.
        if c in set(code_words):
            continue
            
        if can_add_int(c):
            # Add to code
            code_words.append(c)
            
            # Update codebook array
            # We need to append a row
            new_row = np.zeros(17, dtype=np.uint8)
            for i in range(17):
                if (c >> i) & 1:
                    new_row[i] = 1
            codebook = np.vstack([codebook, new_row])
            
            if len(code_words) > len(best_code):
                best_code = code_words[:]
                
    return best_code
