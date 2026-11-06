import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    N = 17
    MIN_DIST = 6
    NUM_WORDS = 1 << N

    # Precompute masks for weights 0 to 5
    masks_by_weight = [[] for _ in range(6)]
    for w in range(6):
        for combo in combinations(range(N), w):
            mask = 0
            for p in combo:
                mask |= (1 << p)
            masks_by_weight[w].append(mask)
    
    # Flatten masks for faster iteration
    # Actually, iterating through lists is fine.
    
    def get_neighbors(c):
        """Yields all words with Hamming distance < MIN_DIST from c."""
        # Dist 0
        yield c
        # Dist 1-5
        for w in range(1, MIN_DIST):
            for mask in masks_by_weight[w]:
                yield c ^ mask

    # 1. Construct the 256 linear code words
    # GF(2^8) arithmetic
    PRIM_POLY = 0x11D
    def gf_mul(a, b):
        res = 0
        for i in range(8):
            if (b >> i) & 1:
                res ^= a
            a <<= 1
            if a & 0x100:
                a ^= PRIM_POLY
        return res

    def gf_pow(a, p):
        res = 1
        for _ in range(p):
            res = gf_mul(res, a)
        return res

    alpha = None
    for a in range(2, 256):
        if gf_pow(a, 17) == 1 and gf_pow(a, 1) != 1:
            alpha = a
            break
            
    if alpha is None:
        # Fallback greedy
        return greedy_fallback(deadline)

    Q = {1, 2, 4, 8, 9, 13, 15, 16}
    g_coeffs = [1]
    for q in Q:
        root = gf_pow(alpha, q)
        new_g = [0] * (len(g_coeffs) + 1)
        for i in range(len(g_coeffs)):
            new_g[i] ^= gf_mul(g_coeffs[i], root)
            new_g[i+1] ^= g_coeffs[i]
        g_coeffs = new_g
        
    bin_g = [1 if c == 1 else 0 for c in g_coeffs]
    while len(bin_g) < 9:
        bin_g.append(0)
        
    g_int = 0
    for i in range(9):
        if bin_g[i]:
            g_int |= (1 << i)
            
    linear_code = set()
    for m in range(512):
        c = 0
        for i in range(9):
            if (m >> i) & 1:
                c ^= (g_int << i)
        if c.bit_count() % 2 == 0:
            linear_code.add(c)
            
    # 2. Search Loop
    best_code_size = len(linear_code)
    best_code = set(linear_code)
    rng = random.Random(42)
    
    all_words = list(range(NUM_WORDS))
    
    while time.time() < deadline - 5.0:
        # Restart with a new random order
        rng.shuffle(all_words)
        
        code_set = set(linear_code)
        forbid_count = np.zeros(NUM_WORDS, dtype=np.int32)
        
        # Initialize forbid_count with linear code
        for c in linear_code:
            for v in get_neighbors(c):
                forbid_count[v] += 1
        
        # Greedy addition
        for w in all_words:
            if time.time() > deadline - 2.0:
                break
            if forbid_count[w] == 0:
                code_set.add(w)
                for v in get_neighbors(w):
                    forbid_count[v] += 1
        
        if len(code_set) > best_code_size:
            best_code_size = len(code_set)
            best_code = set(code_set)
            
        # Optional: Local search on the best found so far?
        # For now, just rely on multiple restarts of the greedy phase.
        # The greedy phase is O(N * 2^N) which is ~ 1.7M * 9400 ~ 16 billion ops?
        # No, the inner loop is only executed if `forbid_count[w] == 0`.
        # Initially, most are forbidden. As we add words, more become forbidden.
        # The number of additions is ~ 300.
        # So total ops ~ 300 * 9400 + 131072 * 1 (check).
        # This is very fast. ~ 3 million ops per restart.
        # We can do hundreds of restarts in 80 seconds.

    return list(best_code)

def greedy_fallback(deadline):
    code = [0]
    code_set = {0}
    rng = random.Random(42)
    N = 17
    while time.time() < deadline:
        w = rng.randint(6, 12)
        positions = rng.sample(range(N), w)
        c = 0
        for p in positions:
            c |= (1 << p)
        if c not in code_set:
            ok = True
            for w2 in code:
                if (c ^ w2).bit_count() < 6:
                    ok = False
                    break
            if ok:
                code.append(c)
                code_set.add(c)
    return code
