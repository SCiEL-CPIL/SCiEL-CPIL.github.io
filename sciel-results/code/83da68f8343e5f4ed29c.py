"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _gf_mul(a, b):
    """Multiply two elements in GF(32) defined by x^5 + x^2 + 1."""
    result = 0
    while b:
        if b & 1:
            result ^= a
        b >>= 1
        a <<= 1
        if a & 32:  # x^5
            a ^= 0b01001  # x^2 + 1
    return result

def _gf_pow(a, n):
    """Raise a to power n in GF(32)."""
    result = 1
    while n:
        if n & 1:
            result = _gf_mul(result, a)
        a = _gf_mul(a, a)
        n >>= 1
    return result

def _poly_mul(p, q):
    """Multiply two polynomials with coefficients in GF(32)."""
    deg_p = len(p) - 1
    deg_q = len(q) - 1
    deg_r = deg_p + deg_q
    r = [0] * (deg_r + 1)
    for i in range(len(p)):
        for j in range(len(q)):
            r[i + j] ^= _gf_mul(p[i], q[j])
    return r

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Parameters for Extended Quadratic Residue Code of length 18 (p=17)
    # Quadratic residues mod 17: {1, 2, 4, 8, 9, 13, 15, 16}
    # These form a single cyclotomic coset C_1 mod 17.
    
    # 1. Compute Generator Polynomial g(x)
    alpha = 2
    coset = [1, 2, 4, 8, 16, 15, 13, 9]
    
    g = [1]
    for i in coset:
        root = _gf_pow(alpha, i)
        factor = [root, 1]
        g = _poly_mul(g, factor)
        
    g_coeffs = [int(c) for c in g]
    
    # 2. Construct Generator Matrix for length 17 code
    n_orig = 17
    k = len(g_coeffs) # 9
    G_orig = []
    for i in range(k):
        row = [0] * n_orig
        for j in range(k):
            row[i + j] = g_coeffs[j]
        G_orig.append(row)
        
    # 3. Extend to length 18
    n_ext = 18
    G_ext = []
    for row in G_orig:
        parity = sum(row) % 2
        G_ext.append(row + [parity])
        
    # 4. Generate all 512 codewords
    words = []
    for m in range(512):
        if time.time() > deadline:
            break
        word = [0] * n_ext
        for i in range(k):
            if (m >> i) & 1:
                for j in range(n_ext):
                    word[j] ^= G_ext[i][j]
        val = 0
        for j in range(n_ext):
            if word[j]:
                val |= (1 << j)
        words.append(val)
        
    return words
# EVOLVE-BLOCK-END
