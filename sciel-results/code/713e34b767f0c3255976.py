import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 60.0

    # GF(2^8) parameters
    IRRED = 0b100011101 # x^8 + x^4 + x^3 + x^2 + 1

    def gf_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= IRRED
        return res

    def gf_pow(a, n):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a)
            a = gf_mul(a, a)
            n >>= 1
        return res

    # Find primitive 17th root of unity
    alpha = None
    for i in range(2, 256):
        if gf_pow(i, 17) == 1 and i != 1:
            alpha = i
            break

    if alpha is None:
        # Fallback: Reed-Muller [16, 5, 8] punctured/extended? 
        # Or just return a small valid code.
        # The seed is 32 words.
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                # Extend to 18 bits? 
                # The seed is 18-bit words.
                # The RM code here is 16-bit.
                # We need 18-bit.
                # Let's just return the 32 words from the seed description if possible, 
                # but we don't have the exact seed code.
                # The current fallback returns 16-bit words. 
                # This might be invalid for N=18.
                # Better to return a simple valid code of size 2 (distance 18) or something.
                # But this branch is unlikely.
                pass
        # Construct a simple [18, 1, 18] code? No, size 2.
        # Or [18, 2, 9]?
        # Let's just return the 32 words from a known construction if we can generate them.
        # For safety, return a list of 2 words: 0 and 2**18-1 (distance 18).
        return [0, (1 << 18) - 1]

    # Compute Minimal Polynomial g(x)
    # Roots: alpha, alpha^2, ..., alpha^128
    poly = [1]
    
    # Precompute powers of alpha
    roots = []
    current = alpha
    for _ in range(8):
        roots.append(current)
        current = gf_mul(current, current) # alpha^(2^j)

    for root in roots:
        new_poly = [0] * (len(poly) + 1)
        for k in range(len(poly)):
            new_poly[k] ^= gf_mul(root, poly[k])
            new_poly[k+1] ^= poly[k]
        poly = new_poly

    # poly has 8 coefficients (degree 7? No, degree 8 means 9 coeffs)
    # Wait, order of 2 mod 17 is 8. So minimal polynomial degree is 8.
    # So poly should have 9 elements.
    
    # Generate Basis
    # g(x) = sum_{j=0}^7 poly[j] x^j ?
    # My loop:
    # Start [1] (deg 0)
    # Mult by (x+root) -> deg 1
    # ...
    # Mult 8 times -> deg 8.
    # So poly has 9 elements.
    
    # Basis vectors: x^i * g(x) for i=0..8
    # Length 17.
    basis = []
    for i in range(9):
        vec = [0] * 17
        for j in range(9):
            if i + j < 17:
                vec[i+j] = poly[j]
        basis.append(vec)
        
    codewords = []
    for mask in range(512):
        word = [0] * 17
        for i in range(9):
            if mask & (1 << i):
                for j in range(17):
                    word[j] ^= basis[i][j]
        
        parity = sum(word) % 2
        val = 0
        for i in range(17):
            if word[i]:
                val |= (1 << i)
        if parity:
            val |= (1 << 17)
        codewords.append(val)
        
    return codewords
