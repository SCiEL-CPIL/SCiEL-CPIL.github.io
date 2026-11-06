import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    # 1. Construct GF(2^8)
    # Primitive polynomial: x^8 + x^4 + x^3 + x^2 + 1 (0x1D)
    # This is a common primitive polynomial for GF(2^8).
    # We need to verify it's primitive. 0x1D is primitive.
    
    def gf_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= 0x1D
        return res

    def gf_exp(a, e):
        res = 1
        while e:
            if e & 1:
                res = gf_mul(res, a)
            a = gf_mul(a, a)
            e >>= 1
        return res

    # 2. Find primitive 17th root of unity alpha in GF(2^8)
    # We need alpha^17 == 1 and alpha != 1
    alpha = None
    for a in range(2, 256):
        if gf_exp(a, 17) == 1:
            alpha = a
            break
    
    if alpha is None:
        # Fallback to seed if something goes wrong
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    # 3. Define Quadratic Residues mod 17
    Q = [1, 2, 4, 8, 9, 13, 15, 16]
    
    # 4. Compute Generator Polynomial g(x) = prod_{i in Q} (x - alpha^i)
    # g(x) is in GF(2^8)[x], but coefficients should end up in GF(2)
    # Start with g(x) = 1
    g_coeffs = [1]  # Represents 1
    
    for i in Q:
        root = gf_exp(alpha, i)
        # Multiply g(x) by (x + root)
        # New degree is len(g_coeffs)
        new_g_coeffs = [0] * (len(g_coeffs) + 1)
        for j in range(len(g_coeffs)):
            # x * g_j x^j = g_j x^{j+1}
            new_g_coeffs[j+1] ^= g_coeffs[j]
            # root * g_j x^j = (root * g_j) x^j
            new_g_coeffs[j] ^= gf_mul(root, g_coeffs[j])
        g_coeffs = new_g_coeffs
        
    # Verify coefficients are 0 or 1
    # g_coeffs length should be 9 (degree 8)
    # Map GF(2^8) values to bits (should be 0 or 1)
    g_bits = []
    for c in g_coeffs:
        if c == 0:
            g_bits.append(0)
        elif c == 1:
            g_bits.append(1)
        else:
            # If not 0 or 1, construction might be flawed for this primitive poly
            # But theoretically it should be in GF(2)
            # For safety, if this happens, fallback
            return construct_fallback()
            
    # 5. Construct Generator Matrix G (9 x 18)
    # Cyclic code length 17 part
    # Vector v of length 17: [g_0, ..., g_8, 0, ..., 0]
    v = g_bits + [0] * (17 - len(g_bits))
    
    G = []
    for k in range(9):
        # Row k is cyclic shift of v to the right by k
        # v_shifted[j] = v[(j - k) % 17]
        row = [0] * 17
        for j in range(17):
            row[j] = v[(j - k) % 17]
        
        # Calculate parity bit
        parity = sum(row) % 2
        row.append(parity)
        
        G.append(row)
        
    # 6. Generate all 512 codewords
    words = []
    for mask in range(512):
        word_int = 0
        for i in range(9):
            if mask & (1 << i):
                # XOR row i into word_int
                # Row i is G[i]
                # Bit j of G[i] corresponds to bit j of word_int
                for j in range(18):
                    if G[i][j]:
                        word_int ^= (1 << j)
        words.append(word_int)
        
    return words

def construct_fallback():
    # Fallback to seed if GF construction fails
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words
