import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    # GF(2^16) parameters
    # Primitive polynomial: x^16 + x^12 + x^3 + x + 1
    # Binary: 1 0000 1000 0000 1011
    # Hex: 0x1100B
    poly = 0x1100B
    
    def gf_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x10000:
                a ^= poly
        return res

    def gf_pow(a, n):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a)
            a = gf_mul(a, a)
            n >>= 1
        return res

    # Alpha is the root of the primitive polynomial, represented as 2 (x)
    alpha = 2

    # Quadratic residues mod 17
    Q = [1, 4, 9, 16, 8, 2, 15, 13]
    
    # Compute generator polynomial g(x) = prod_{i in Q} (x - alpha^i)
    # g(x) = x^8 + c7 x^7 + ... + c0
    g_coeffs = [1]
    
    for i in Q:
        root = gf_pow(alpha, i)
        new_g = [0] * (len(g_coeffs) + 1)
        for j in range(len(g_coeffs)):
            new_g[j+1] ^= g_coeffs[j]
            new_g[j] ^= gf_mul(g_coeffs[j], root)
        g_coeffs = new_g
        
    # g_coeffs length 9
    # Convert to binary
    g_bin = [1 if c in (0, 1) else 0 for c in g_coeffs]
    
    words = []
    # 512 messages (9 bits)
    for m in range(512):
        p_coeffs = [0] * 17
        for i in range(9):
            if (m >> i) & 1:
                for j in range(9):
                    if g_bin[j]:
                        p_coeffs[(i+j) % 17] ^= 1
                        
        w = 0
        for k in range(17):
            if p_coeffs[k]:
                w |= (1 << k)
                
        # Even weight filter
        if w.bit_count() % 2 == 0:
            words.append(w)
            
    return words
