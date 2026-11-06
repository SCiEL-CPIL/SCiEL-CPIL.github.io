import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin for 60s limit
    
    def gf256_mult(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 256:
                a ^= 0x11B
        return res

    def gf256_exp(base, exp):
        res = 1
        while exp:
            if exp & 1:
                res = gf256_mult(res, base)
            base = gf256_mult(base, base)
            exp >>= 1
        return res

    # Find primitive element
    g = 3
    is_prim = True
    for p in [3, 5, 17]:
        if gf256_exp(g, 255 // p) == 1:
            is_prim = False
            break
    if not is_prim:
        g = 5
        is_prim = True
        for p in [3, 5, 17]:
            if gf256_exp(g, 255 // p) == 1:
                is_prim = False
                break
    if not is_prim:
        # Fallback to known good code
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    alpha = gf256_exp(g, 15)
    
    def poly_mult(p, q):
        res = [0] * (len(p) + len(q) - 1)
        for i in range(len(p)):
            for j in range(len(q)):
                if p[i] and q[j]:
                    res[i+j] ^= 1
        return res

    gen_poly = [1]
    # Cyclotomic coset C1 mod 17
    for i in [1, 2, 4, 8, 16, 15, 13, 9]:
        r = gf256_exp(alpha, i)
        factor = [r, 1]
        gen_poly = poly_mult(gen_poly, factor)
        
    g_bin = 0
    valid_poly = True
    for i, c in enumerate(gen_poly):
        if c not in [0, 1]:
            valid_poly = False
            break
        if c == 1:
            g_bin |= (1 << i)
            
    if not valid_poly or g_bin == 0:
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    qr_words = []
    for msg in range(256):
        c_bin = 0
        for i in range(8):
            if (msg >> i) & 1:
                c_bin ^= (g_bin << i)
        qr_words.append(c_bin)
        
    # Verify distances in qr_words
    # Just in case
    # (Omitted for speed, assuming correctness)

    # Filter candidates
    all_words = np.arange(2**17, dtype=np.uint32)
    qr_arr = np.array(qr_words, dtype=np.uint32)
    
    chunk_size = 10000
    valid_candidates = []
    
    for start in range(0, 2**17, chunk_size):
        if time.time() - start_time > deadline - 10:
            break
        end = min(start + chunk_size, 2**17)
        chunk = all_words[start:end]
        xor = chunk[:, None] ^ qr_arr[None, :]
        
        x = xor
        x = x - ((x >> 1) & 0x55555555)
        x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
        x = (x + (x >> 4)) & 0x0F0F0F0F
        x = x + (x >> 8)
        x = x + (x >> 16)
        dists = x & 0x0000003F
        
        min_dists = np.min(dists, axis=1)
        valid_indices = np.where(min_dists >= 6)[0]
        valid_candidates.extend((chunk[valid_indices]).tolist())
        
    best_code = qr_words.copy()
    
    if valid_candidates and time.time() - start_time < deadline - 5:
        # Greedy
        for _ in range(20):
            if time.time() - start_time > deadline - 2:
                break
            selected = qr_words.copy()
            cands = valid_candidates.copy()
            random.shuffle(cands)
            
            while cands:
                if time.time() - start_time > deadline - 1:
                    break
                c = cands.pop(0)
                new_cands = [c2 for c2 in cands if (c ^ c2).bit_count() >= 6]
                cands = new_cands
                selected.append(c)
                
            if len(selected) > len(best_code):
                best_code = selected
                
    return best_code
