"""Construct binary error-correcting code of length 18, distance 6 via Extended QR Code."""
import random
import numpy as np
from itertools import combinations

N = 18
MIN_DIST = 6
MAX_WORD = 1 << N

def popcount(x):
    return bin(x).count('1')

def hamming(x, y):
    return popcount(x ^ y)

def gf_mult(a, b, poly=0x11D):
    res = 0
    while b:
        if b & 1:
            res ^= a
        b >>= 1
        a <<= 1
        if a & 0x100:
            a ^= poly
    return res

def gf_pow(a, n, poly=0x11D):
    res = 1
    while n:
        if n & 1:
            res = gf_mult(res, a, poly)
        a = gf_mult(a, a, poly)
        n >>= 1
    return res

def poly_mul_gf(p, q, poly=0x11D):
    res = [0] * (len(p) + len(q) - 1)
    for i, pi in enumerate(p):
        if pi == 0: continue
        for j, qj in enumerate(q):
            if qj == 0: continue
            res[i+j] ^= gf_mult(pi, qj, poly)
    return res

def min_dist_to_set(word, code):
    """Return minimum Hamming distance from word to any codeword in code."""
    md = N + 1
    for c in code:
        d = popcount(word ^ c)
        if d < md:
            md = d
            if md < MIN_DIST:
                break
    return md

def greedy_add(code, word):
    """Try to add word to code, return True if successful."""
    for c in code:
        if popcount(word ^ c) < MIN_DIST:
            return False
    code.append(word)
    return True

def local_search_improve(code, max_iters=200):
    """Try to add more codewords via local search: mutate random words and try to add."""
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        # Try to add a new codeword by mutating a random base
        base = random.randint(0, (1 << N) - 1)
        # Try adding base directly
        if min_dist_to_set(base, code) >= MIN_DIST:
            code.append(base)
            improved = True
            continue
        # Try single-bit flips
        for i in range(N):
            w = base ^ (1 << i)
            if min_dist_to_set(w, code) >= MIN_DIST:
                code.append(w)
                improved = True
                break
        if improved:
            continue
        # Try double-bit flips
        for i in range(N):
            for j in range(i+1, N):
                w = base ^ (1 << i) ^ (1 << j)
                if min_dist_to_set(w, code) >= MIN_DIST:
                    code.append(w)
                    improved = True
                    break
            if improved:
                break
    return code

def greedy_construction(seed, order_type='random'):
    """Greedy construction with specified ordering."""
    rng = random.Random(seed)
    code = [0]
    
    # Generate candidates
    if order_type == 'random':
        candidates = list(range(1, 1 << N))
        rng.shuffle(candidates)
    elif order_type == 'weight_asc':
        # Sort by weight ascending
        candidates = sorted(range(1, 1 << N), key=popcount)
        # Shuffle within weights? Let's keep sorted for determinism in this pass
        # Actually, shuffling within weight buckets is better
        weights = {}
        for w in range(1, 1 << N):
            p = popcount(w)
            if p not in weights:
                weights[p] = []
            weights[p].append(w)
        candidates = []
        for p in sorted(weights.keys()):
            rng.shuffle(weights[p])
            candidates.extend(weights[p])
    elif order_type == 'weight_desc':
        weights = {}
        for w in range(1, 1 << N):
            p = popcount(w)
            if p not in weights:
                weights[p] = []
            weights[p].append(w)
        candidates = []
        for p in sorted(weights.keys(), reverse=True):
            rng.shuffle(weights[p])
            candidates.extend(weights[p])
    elif order_type == 'weight_mid':
        # Start from middle weight (9) and go out
        weights = {}
        for w in range(1, 1 << N):
            p = popcount(w)
            if p not in weights:
                weights[p] = []
            weights[p].append(w)
        
        # Order: 9, 8, 10, 7, 11, 6, 12, ...
        order = [9, 8, 10, 7, 11, 6, 12, 5, 13, 4, 14, 3, 15, 2, 16, 1, 17]
        candidates = []
        for p in order:
            if p in weights:
                rng.shuffle(weights[p])
                candidates.extend(weights[p])
        
    for w in candidates:
        if min_dist_to_set(w, code) >= MIN_DIST:
            code.append(w)
            if len(code) >= 512:
                break
    return code

def refine_code(code, max_iters=100):
    """Try to add more words by iterating through a shuffled list of all words."""
    rng = random.Random(12345)
    candidates = list(range(1 << N))
    rng.shuffle(candidates)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for w in candidates:
            if min_dist_to_set(w, code) >= MIN_DIST:
                code.append(w)
                improved = True
                if len(code) >= 512:
                    return code
    return code

def construct():
    """Construct the largest possible binary code of length 18, distance 6."""
    # 1. Find generator polynomial g(x) for [17, 9, 5] QR code
    # Use GF(2^8) with poly 0x11D
    beta = 2
    alpha = gf_pow(beta, 15, 0x11D)
    
    # Calculate minimal polynomial m1(x) = prod (x + alpha^(2^j))
    m1 = [1] # 1
    for j in range(8):
        conj = gf_pow(alpha, (1 << j), 0x11D)
        factor = [conj, 1]
        m1 = poly_mul_gf(m1, factor, 0x11D)
        
    # Convert m1 to integer g_int
    g_int = 0
    for i, c in enumerate(m1):
        if c & 1:
            g_int |= (1 << i)
            
    # 2. Generate 512 codewords
    codes = []
    for msg in range(512):
        c_bits = 0
        for i in range(9):
            if (msg >> i) & 1:
                c_bits ^= (g_int << i)
                
        # Extend to 18 bits with parity
        parity = popcount(c_bits) & 1
        full_word = c_bits | (parity << 17)
        codes.append(full_word)
        
    return codes
