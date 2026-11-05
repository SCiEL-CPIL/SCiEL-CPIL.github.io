import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    # 1. GF(2^8) setup
    # Primitive polynomial x^8 + x^4 + x^3 + x^2 + 1
    P = 0x1D
    
    def gf_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= P
        return res

    def gf_pow(a, n):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a)
            a = gf_mul(a, a)
            n >>= 1
        return res

    # 2. Find primitive 17th root of unity
    # beta is x (0x02)
    beta = 0x02
    alpha = gf_pow(beta, 15)
    
    # 3. Compute generator polynomial g(x)
    # g(x) = prod_{j=0}^{7} (x - alpha^(2^j))
    # Coeffs in GF(2^8)
    g_coeffs = [1]
    root = alpha
    for _ in range(8):
        new_g = [0] * (len(g_coeffs) + 1)
        for i, c in enumerate(g_coeffs):
            new_g[i] ^= gf_mul(c, root)
            new_g[i+1] ^= c
        g_coeffs = new_g
        root = gf_mul(root, root)
        
    # Convert to bitmask over F2
    g_poly = 0
    for i, c in enumerate(g_coeffs):
        if c == 1:
            g_poly |= (1 << i)
        elif c != 0:
            # Should not happen for QR code over F2
            pass
            
    # 4. Generate codewords
    words = []
    # Dimension 9, so 512 codewords
    # g_poly has degree 8
    deg_g = g_poly.bit_length() - 1
    
    def poly_mod(a, g):
        # a is bitmask, g is bitmask
        # Returns a mod g
        # Assumes deg(a) >= deg(g) initially or handles it
        while a.bit_length() > deg_g + 1: # a has degree >= deg_g
            shift = a.bit_length() - (deg_g + 1)
            a ^= (g << shift)
        return a

    for m in range(512):
        # m is 9 bits
        # u(x) = x^8 * m(x)
        u = m << 8
        r = poly_mod(u, g_poly)
        c = u ^ r
        # c is 17 bits
        # Check parity
        if c.bit_count() % 2 == 0:
            words.append(c)
            
    # Optional: Verify min distance
    # With 256 words, this is fast
    # But we trust the construction.
    
    return words
