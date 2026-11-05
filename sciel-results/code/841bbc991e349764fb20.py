import time

N = 18
MIN_DISTANCE = 6

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # GF(2^8) arithmetic
    # Primitive polynomial for GF(2^8): x^8 + x^4 + x^3 + x^2 + 1 (0x11D)
    # Or standard AES poly: x^8 + x^4 + x^3 + x + 1 (0x11B)
    # We need a field containing a primitive 17th root of unity.
    # Order of element must divide 255. 17 divides 255.
    
    def gf28_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= 0x11D  # Primitive poly x^8+x^4+x^3+x^2+1
        return res

    def gf28_pow(a, n):
        res = 1
        while n:
            if n & 1:
                res = gf28_mul(res, a)
            a = gf28_mul(a, a)
            n >>= 1
        return res

    # Find primitive element beta
    # Try small integers
    beta = 2
    # Check if beta is primitive (order 255)
    # 255 = 3 * 5 * 17
    # beta^15 != 1, beta^51 != 1, beta^15 != 1
    
    # Compute alpha = beta^15 (order 17)
    alpha = gf28_pow(beta, 15)
    
    # Compute minimal polynomial of alpha
    # Roots are alpha^1, alpha^2, alpha^4, alpha^8, alpha^16, alpha^15, alpha^13, alpha^9
    # g(x) = (x - alpha)(x - alpha^2)...(x - alpha^9)
    # Coefficients in GF(2^8), but they should be in GF(2)
    
    # Polynomial multiplication in GF(2)[x]
    # Represent poly as list of coefficients [c0, c1, ...]
    def poly_mul(a, b):
        res = [0] * (len(a) + len(b) - 1)
        for i, ai in enumerate(a):
            for j, bj in enumerate(b):
                res[i+j] ^= gf28_mul(ai, bj)
        return res

    # Start with 1
    g = [1]
    for i in range(8):
        # Root is alpha^(2^i)
        root = gf28_pow(alpha, 1 << i)
        # Factor is (x - root) = (x + root) in GF(2)
        # Coeffs: [root, 1]
        g = poly_mul(g, [root, 1])
        
    # g should have coefficients 0 or 1
    # Verify
    for c in g:
        if c not in [0, 1]:
            # Fallback if field arithmetic is wrong
            break
            
    if any(c not in [0, 1] for c in g):
        # Use hardcoded coefficients from seed if derivation fails
        g_indices = [0, 3, 4, 5, 8]
        g_coeffs = [0]*9
        for idx in g_indices:
            g_coeffs[idx] = 1
    else:
        g_coeffs = g
        
    # g_coeffs is [c0, c1, ..., c8] for c0 + c1 x + ... + c8 x^8
    # Degree is 8.
    
    # Build generator matrix rows
    # The code is cyclic.
    # Generator vector is [g0, g1, ..., g8, 0, ..., 0]
    # Shifts give other rows.
    # Dimension is 17 - 8 = 9.
    
    gen_matrix = []
    for i in range(9):
        row = [0] * 17
        for j in range(9):
            if g_coeffs[j] == 1:
                pos = (i + j) % 17
                row[pos] = 1
        # Extend to length 18 with parity
        parity = sum(row) % 2
        new_row = row + [parity]
        
        val = 0
        for k in range(18):
            if new_row[k]:
                val |= (1 << k)
        gen_matrix.append(val)
        
    # Generate all 512 codewords
    code_words = [0]
    for g in gen_matrix:
        code_words = code_words + [w ^ g for w in code_words]
        
    # Verify min distance
    min_d = 19
    for w in code_words:
        if w != 0:
            wt = w.bit_count()
            if wt < min_d:
                min_d = wt
            if min_d < MIN_DISTANCE:
                break
                
    if min_d >= MIN_DISTANCE:
        return code_words
        
    # Fallback: Reed-Muller RM(1,4)
    rm_basis = []
    b0 = 0
    for i in range(16):
        b0 |= 1 << i
    rm_basis.append(b0)
    for j in range(4):
        bj = 0
        for i in range(16):
            if (i >> j) & 1:
                bj |= 1 << i
        rm_basis.append(bj)
        
    code_words_rm = [0]
    for b in rm_basis:
        code_words_rm = code_words_rm + [w ^ b for w in code_words_rm]
        
    return code_words_rm
