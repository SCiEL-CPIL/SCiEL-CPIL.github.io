import numpy as np
from itertools import combinations

def construct():
    # Step 1: Generate the 512 codewords of the [17, 9, 5] QR code, extended to 18
    # Reuse the logic from the seed program to get the base code
    
    def poly_mod(a, b):
        while b != 0:
            deg_a = a.bit_length() - 1
            deg_b = b.bit_length() - 1
            if deg_a >= deg_b:
                shift = deg_a - deg_b
                a ^= (b << shift)
            else:
                break
        return a

    def poly_divides(a, b):
        return poly_mod(a, b) == 0

    def is_irreducible(p, n):
        def poly_pow_mod(base, exp, mod):
            result = 1
            base = poly_mod(base, mod)
            while exp:
                if exp & 1:
                    result = poly_mod(result * base, mod)
                base = poly_mod(base * base, mod)
                exp >>= 1
            return result

        x = 2
        val = x
        for _ in range(n):
            val = poly_mod(val * val, p)
        if poly_mod(val ^ x, p) != 0:
            return False
        
        for k in [1, 2, 4]:
            val = x
            for _ in range(k):
                val = poly_mod(val * val, p)
            if poly_mod(val ^ x, p) == 0:
                return False
        return True

    Q = 0
    for i in range(9):
        Q |= 1 << (2*i)
        
    g = None
    for p in range(1 << 8, 1 << 9):
        if p & 1 == 0:
            continue
        if is_irreducible(p, 8):
            if poly_divides(Q, p):
                g = p
                break
                
    if g is None:
        # Fallback
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    words_17 = []
    for m in range(512):
        c = 0
        for i in range(9):
            if (m >> i) & 1:
                c ^= g << i
        words_17.append(c)
        
    base_words = []
    for c in words_17:
        parity = c.bit_count() % 2
        c_ext = c | (parity << 17)
        base_words.append(c_ext)
        
    # Step 2: Greedy Expansion
    N = 18
    MIN_D = 6
    R = MIN_D - 1 # 5
    
    # Generate all masks of weight <= R
    masks = [0]
    for r in range(1, R + 1):
        for comb in combinations(range(N), r):
            mask = 0
            for i in comb:
                mask |= 1 << i
            masks.append(mask)
            
    # Use a set or boolean array for forbidden words
    # Boolean array is faster for lookup
    forbidden = bytearray(1 << N)
    
    # Mark neighborhoods of base words
    for w in base_words:
        for m in masks:
            forbidden[w ^ m] = 1
            
    # Greedy add
    new_words = list(base_words)
    for x in range(1 << N):
        if not forbidden[x]:
            # Add x
            new_words.append(x)
            # Mark its neighborhood
            for m in masks:
                forbidden[x ^ m] = 1
                
    return new_words
