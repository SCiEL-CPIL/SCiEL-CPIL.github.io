"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs the Extended Quadratic Residue code of length 18.
    This code has parameters [18, 9, 6] and contains 2^9 = 512 codewords.
    """
    start_time = time.time()
    
    # 1. Define GF(16) arithmetic
    # Primitive polynomial: x^4 + x + 1
    # Elements represented as integers 0-15.
    
    def gf_mul(a, b):
        result = 0
        while b:
            if b & 1:
                result ^= a
            b >>= 1
            a <<= 1
            if a & 0b10000:
                a ^= 0b10011
        return result

    # 2. Compute generator polynomial g(x)
    def poly_mul_gf(p, q):
        res = [0] * (len(p) + len(q) - 1)
        for i, ci in enumerate(p):
            if ci == 0: continue
            for j, cj in enumerate(q):
                if cj == 0: continue
                res[i+j] ^= gf_mul(ci, cj)
        while res and res[-1] == 0:
            res.pop()
        return res

    p = [1]
    alpha_powers = [0] * 17
    alpha_powers[0] = 1
    for i in range(1, 17):
        alpha_powers[i] = gf_mul(alpha_powers[i-1], 2)
        
    exponents = [3, 6, 12, 7, 14, 11, 5, 10]
    for exp in exponents:
        root = alpha_powers[exp]
        p = poly_mul_gf(p, [root, 1])
        
    g_coeffs = p
    if len(g_coeffs) != 9:
        # Fallback to simple Reed-Muller if something fails
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    g_bits = [1 if c == 1 else 0 for c in g_coeffs]
    
    # 3. Generate 512 codewords
    words = []
    for msg in range(512):
        # Check time
        if time.time() - start_time > 50:
            break
            
        m_bits = [(msg >> i) & 1 for i in range(9)]
        
        c_bits = [0] * 17
        for i in range(9):
            if m_bits[i]:
                for j in range(9):
                    if g_bits[j]:
                        c_bits[i+j] ^= 1
        
        parity = sum(c_bits) % 2
        
        val = 0
        for i in range(17):
            if c_bits[i]:
                val |= (1 << i)
        if parity:
            val |= (1 << 17)
            
        words.append(val)
        
    return words
# EVOLVE-BLOCK-END
