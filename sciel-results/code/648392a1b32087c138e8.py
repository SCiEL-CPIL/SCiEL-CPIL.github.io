import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 17
    MIN_DIST = 6
    
    def time_left():
        return deadline - time.time()

    # 1. Construct the [17,8,6] QR code
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

    # Find alpha (primitive element)
    alpha = None
    for a in range(2, 256):
        if gf_pow(a, 17) == 1 and gf_pow(a, 1) != 1:
            alpha = a
            break
            
    if alpha is None:
        return greedy_construct(deadline)

    # Quadratic residues mod 17
    Q = {1, 2, 4, 8, 9, 13, 15, 16}
    
    # Calculate generator polynomial g(x) = prod_{q in Q} (x + alpha^q)
    g_coeffs = [1]
    
    for q in Q:
        root = gf_pow(alpha, q)
        new_g = [0] * (len(g_coeffs) + 1)
        for i in range(len(g_coeffs)):
            new_g[i] ^= gf_mul(g_coeffs[i], root)
            new_g[i+1] ^= g_coeffs[i]
        g_coeffs = new_g
        
    # Convert to binary polynomial coefficients
    # Coeffs should be 0 or 1
    bin_g = [1 if c == 1 else 0 for c in g_coeffs]
    
    # Ensure length 9 (degree 8)
    while len(bin_g) < 9:
        bin_g.append(0)
        
    # Convert bin_g to integer for fast shifting
    g_int = 0
    for i in range(9):
        if bin_g[i]:
            g_int |= (1 << i)
            
    # Generate all 256 codewords of the [17,8,6] code
    base_code = []
    
    for m in range(256):
        c = 0
        for i in range(8):
            if (m >> i) & 1:
                c ^= (g_int << i)
        
        # c is a 17-bit word
        base_code.append(c)
        
    code = base_code[:]
    code_set = set(code)
    
    # Fast Greedy Expansion
    # Using Python's bit_count() is efficient.
    
    def can_add(cand):
        if cand in code_set:
            return False
        for w in code:
            if (cand ^ w).bit_count() < MIN_DIST:
                return False
        return True

    rng = random.Random(42)
    
    # Generate candidates for weights 6-10
    candidates = []
    for w in range(6, 11):
        for combo in combinations(range(N), w):
            c = 0
            for p in combo:
                c |= (1 << p)
            candidates.append(c)
            
    rng.shuffle(candidates)
    
    # Pass 1: Systematic addition
    for c in candidates:
        if time_left() < 5.0:
            break
        if can_add(c):
            code.append(c)
            code_set.add(c)
            
    # Pass 2: Random attempts
    for _ in range(100000):
        if time_left() < 5.0:
            break
        w = rng.randint(6, 12)
        positions = rng.sample(range(N), w)
        c = 0
        for p in positions:
            c |= (1 << p)
        if can_add(c):
            code.append(c)
            code_set.add(c)
            
    # Pass 3: Second systematic pass on remaining candidates
    remaining = [c for c in candidates if c not in code_set]
    rng.shuffle(remaining)
    for c in remaining:
        if time_left() < 5.0:
            break
        if can_add(c):
            code.append(c)
            code_set.add(c)
            
    # Local Search: Try to improve by swapping
    # Simple 1-for-1 swap doesn't increase size.
    # We need to find a way to add more.
    # Let's try random restarts of the greedy on the current code? No, that's expensive.
    
    # Try to add more random words
    for _ in range(50000):
        if time_left() < 2.0:
            break
        w = rng.randint(6, 12)
        positions = rng.sample(range(N), w)
        c = 0
        for p in positions:
            c |= (1 << p)
        if can_add(c):
            code.append(c)
            code_set.add(c)
            
    return code

def greedy_construct(deadline):
    code = [0]
    code_set = {0}
    rng = random.Random(42)
    while time.time() < deadline - 1:
        w = rng.randint(3, 14)
        positions = rng.sample(range(17), w)
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
