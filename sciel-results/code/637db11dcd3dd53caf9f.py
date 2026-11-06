"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    """
    Construct a large binary code of length 18 with minimum distance 6.
    Uses the 512-word extended quadratic residue code as a seed.
    """
    deadline = time.time() + 55.0
    N = 18
    MIN_DIST = 6
    
    # Helper to generate GF(2^8)
    # Primitive polynomial: x^8 + x^4 + x^3 + x^2 + 1
    # We represent elements as integers 0-255
    # Multiplication in GF(2^8)
    def gf_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= 0x11D # 0x100 + 0x14 + 0x08 + 0x04 + 0x01 = 0x11D
        return res

    def gf_pow(a, n):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a)
            a = gf_mul(a, a)
            n >>= 1
        return res

    # Find an element of order 17
    # 17 is prime. We need beta^17 = 1.
    # Since 2^8 = 256 = 15*17 + 1, the order divides 255.
    # We look for an element in GF(2^8) \ {0, 1} such that beta^17 == 1.
    beta = None
    for candidate in range(2, 256):
        if gf_pow(candidate, 17) == 1:
            # Check it's not 1
            if candidate != 1:
                beta = candidate
                break
    
    if beta is None:
        # Fallback to Reed-Muller if something goes wrong
        def reed_muller_seed():
            words = []
            for slope in range(16):
                for intercept in range(2):
                    word = 0
                    for point in range(16):
                        bit = ((slope & point).bit_count() + intercept) % 2
                        word |= bit << point
                    words.append(word)
            return words
        return reed_muller_seed()

    # Compute minimal polynomial of beta
    # g(x) = x^8 + c7 x^7 + ... + c0
    # beta^8 + c7 beta^7 + ... + c0 = 0
    # c0 is 1 because beta is non-zero and poly is monic? No, constant term is 1 for minimal poly of non-zero element.
    # We solve for c0..c7
    # beta^8 = c7 beta^7 + ... + c0
    # This is a linear system over GF(2).
    # Actually, we can just compute the polynomial by finding the smallest degree poly.
    # Since we know degree is 8, we can determine coefficients.
    # Let g(x) = sum_{i=0}^8 g_i x^i with g_8=1.
    # We need sum g_i beta^i = 0.
    # This is a system of 8 equations (coeffs of beta^0..beta^7) for 8 unknowns g_0..g_7.
    # Matrix A where A[j][i] is the coefficient of beta^j in beta^i? No.
    # It's easier to just iterate through all 2^8=256 polynomials of degree 8? No, 2^7=128 monic polys.
    # Check which one has beta as a root.
    g_poly = None
    for coeffs in range(256):
        # coeffs represents g_0..g_7
        # g(x) = x^8 + g_7 x^7 + ... + g_0
        val = 1 # beta^8
        # We need to evaluate sum g_i beta^i
        # Wait, g(beta) = beta^8 + g_7 beta^7 + ... + g_0
        # We can compute this
        total = gf_pow(beta, 8)
        for i in range(8):
            if (coeffs >> i) & 1:
                total ^= gf_pow(beta, i)
        if total == 0:
            g_poly = coeffs
            break
            
    if g_poly is None:
        # Fallback
        def reed_muller_seed():
            words = []
            for slope in range(16):
                for intercept in range(2):
                    word = 0
                    for point in range(16):
                        bit = ((slope & point).bit_count() + intercept) % 2
                        word |= bit << point
                    words.append(word)
            return words
        return reed_muller_seed()

    # Generate the 9 basis vectors for the [17, 9, 5] code
    # The generator matrix rows are g(x), xg(x), ..., x^8 g(x) mod (x^17 - 1)
    # We represent polynomials as bitmasks of length 17.
    def poly_to_mask(poly_coeffs, degree, length):
        # poly_coeffs is the integer representation of the polynomial (LSB is coeff of x^0)
        # This is tricky because gf_poly is g_0..g_7.
        # g(x) = x^8 + g_7 x^7 + ... + g_0
        # Mask bits 0..16
        mask = 0
        # x^8 term
        mask |= (1 << 8)
        for i in range(8):
            if (g_poly >> i) & 1:
                mask |= (1 << i)
        return mask

    g_mask = poly_to_mask(g_poly, 8, 17)
    
    basis = []
    current = g_mask
    for i in range(9):
        basis.append(current)
        # Multiply by x mod (x^17 - 1)
        # Shift left by 1, and if bit 16 is set, XOR with 1 (since x^17 = 1)
        current = (current << 1) & 0x1FFFF
        if current & (1 << 17):
            current ^= 1
        current &= 0x1FFFF

    # Extend to length 18
    # Add parity bit to make weight even.
    # Original min dist 5 (odd). Extension makes it 6 (even).
    code = []
    for mask in basis:
        # Generate all 512 linear combinations?
        # No, the code is the span of these 9 vectors.
        # We need to generate all 2^9 = 512 words.
        pass

    # Generate all 512 words
    words_17 = []
    for mask in range(512):
        w = 0
        for i in range(9):
            if (mask >> i) & 1:
                w ^= basis[i]
        words_17.append(w)
        
    # Extend to 18
    words_18 = []
    for w in words_17:
        weight = w.bit_count()
        parity = weight % 2
        # If weight is odd, parity bit is 1 to make it even?
        # Distance between two even-weight words is even.
        # Original dist is 5. New dist is 5 + |p1-p2|?
        # If w1, w2 have odd weight, p1=1, p2=1. Dist is 5.
        # Wait. If we add parity bit such that total weight is even:
        # If w has odd weight, p=1. Total weight even.
        # If w has even weight, p=0. Total weight even.
        # Distance between two extended words:
        # d(w1, w2) = d(w1_ext, w2_ext).
        # If w1, w2 both odd weight: p1=1, p2=1. Dist = d17 + 0 = 5.
        # This is NOT 6.
        # Standard extension: Append a bit such that the total weight is even.
        # For a code with min distance 5, the extended code has min distance 6?
        # Yes, if the original code has min distance 5, and we extend by parity,
        # the distance between any two distinct codewords increases by 0 or 1.
        # If d(u,v) is odd, the parity bits must be different (one even weight, one odd weight in original? No).
        # Let's check:
        # u, v in C. d(u,v) = 5.
        # w(u) + w(v) = w(u+v) = 5 (odd).
        # So one of u, v has even weight, one has odd weight.
        # Parity bit for even weight is 0. Parity bit for odd weight is 1.
        # So p_u != p_v.
        # d(u_ext, v_ext) = d(u,v) + 1 = 6.
        # So yes, extending with parity increases distance by 1 for all pairs at odd distance.
        # Since min dist is 5, all pairs are at dist 5 or more.
        # Are there pairs at dist 7?
        # If d(u,v)=7, w(u+v)=7 (odd). One even, one odd. p_u != p_v.
        # d_ext = 7+1=8.
        # So min dist becomes 6.
        
        p = weight % 2
        w_ext = w | (p << 17)
        words_18.append(w_ext)
        
    best_code = words_18
    best_len = len(best_code)
    
    # Local search to try to exceed 512
    # Try to add words at distance >= 6 from all existing codewords
    import random
    rng = random.Random(42)
    
    # Precompute all codewords as a list for fast access
    code_set = set(best_code)
    code_list = list(best_code)
    
    # Greedy addition: try random 18-bit words
    attempts = 0
    max_attempts = 500000
    
    while time.time() < deadline - 5.0 and attempts < max_attempts:
        attempts += 1
        # Generate random candidate
        candidate = rng.randint(0, (1 << 18) - 1)
        if candidate in code_set:
            continue
        
        # Check distance to all existing codewords
        valid = True
        for cw in code_list:
            if (candidate ^ cw).bit_count() < MIN_DIST:
                valid = False
                break
        
        if valid:
            code_list.append(candidate)
            code_set.add(candidate)
    
    # If we found additional words, try a few more rounds with different seeds
    if len(code_list) > best_len:
        for seed in [123, 456, 789]:
            rng2 = random.Random(seed)
            for _ in range(200000):
                if time.time() > deadline - 5.0:
                    break
                candidate = rng2.randint(0, (1 << 18) - 1)
                if candidate in code_set:
                    continue
                valid = True
                for cw in code_list:
                    if (candidate ^ cw).bit_count() < MIN_DIST:
                        valid = False
                        break
                if valid:
                    code_list.append(candidate)
                    code_set.add(candidate)
    
    # Use the larger code
    if len(code_list) > len(best_code):
        best_code = code_list
    
    return best_code

# EVOLVE-BLOCK-END
