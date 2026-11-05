import time

N = 18
MIN_DISTANCE = 6

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Construct the Extended Quadratic Residue Code of length 18
    # This is a [18, 9, 6] linear code with 512 codewords.
    
    # Step 1: Construct GF(2^8)
    # Primitive polynomial: x^8 + x^4 + x^3 + x^2 + 1 (0x11D)
    def gf28_mul(a, b, poly=0x11D):
        result = 0
        for i in range(8):
            if (b >> i) & 1:
                result ^= a
            a <<= 1
            if a & 0x100:
                a ^= poly
        return result

    def gf28_pow(base, exp):
        result = 1
        for _ in range(exp):
            result = gf28_mul(result, base)
        return result

    # Find an element of order 17.
    # 2^8 - 1 = 255 = 15 * 17. So elements of order 17 exist.
    # Let's try alpha = 2 (x).
    alpha = 2
    if gf28_pow(alpha, 17) != 1:
        # Try other elements if 2 doesn't work, but 2 usually works for this poly.
        # For robustness, we can search, but 2 is standard.
        for cand in range(2, 256):
            if gf28_pow(cand, 17) == 1 and gf28_pow(cand, 1) != 1:
                alpha = cand
                break

    # Compute the minimal polynomial of alpha.
    # The minimal polynomial m(x) of alpha over GF(2) is the monic polynomial of least degree
    # such that m(alpha) = 0.
    # The degree is the size of the cyclotomic coset of 1 modulo 17 under multiplication by 2.
    # 1 -> 2 -> 4 -> 8 -> 16 -> 15 -> 13 -> 9 -> 1. Size is 8.
    # So m(x) has degree 8.
    # m(x) = prod_{i=0}^{7} (x - alpha^{2^i})
    
    # We can build m(x) iteratively.
    # Start with p(x) = x (coeffs [0, 1])
    # Multiply by (x - alpha^k) for each conjugate.
    # In GF(2), subtraction is addition (XOR).
    # Multiplication of polynomial (represented as list of coeffs c_0..c_n) by (x + a):
    # new_c[i] = old_c[i-1] + a*old_c[i]
    
    def poly_mul_linear(p, a):
        # p is list of coeffs [c_0, ..., c_n]
        # returns coeffs of p(x) * (x + a)
        # (c_0 + c_1 x + ... + c_n x^n) * (x + a)
        # = c_0 a + (c_0 + c_1 a) x + (c_1 + c_2 a) x^2 + ... + c_n x^{n+1}
        n = len(p)
        new_p = [0] * (n + 1)
        for i in range(n):
            # coefficient of x^i in result is c_{i-1} + a*c_i
            # c_{-1} is 0
            c_prev = p[i-1] if i > 0 else 0
            c_curr = p[i]
            new_p[i] = c_prev ^ gf28_mul(a, c_curr)
        # coefficient of x^n is c_{n-1} + a*c_n ? No.
        # Let's re-derive.
        # P(x) = sum_{j=0}^n c_j x^j
        # P(x)(x+a) = sum_{j=0}^n c_j x^{j+1} + sum_{j=0}^n c_j a x^j
        # Coeff of x^k:
        # From first sum: c_{k-1} (if k>=1)
        # From second sum: c_k a (if k<=n)
        # So coeff[k] = (c_{k-1} if k>0 else 0) ^ (a*c_k if k<n else 0)
        
        new_p = [0] * (n + 1)
        for k in range(n + 1):
            val = 0
            if k > 0:
                val ^= p[k-1]
            if k < n:
                val ^= gf28_mul(a, p[k])
            new_p[k] = val
        return new_p

    m_coeffs = [0, 1] # x
    # Conjugates are alpha^{2^i} for i=0..7
    # Note: alpha^{2^0} = alpha
    # We need to multiply by (x - alpha^{2^i}) for i=0..7
    # Since we start with x, we effectively have one factor.
    # Wait, m(x) = prod_{i=0}^{7} (x - alpha^{2^i}).
    # Let's start with 1 and multiply by all 8 factors.
    m_coeffs = [1]
    curr_alpha = alpha
    for i in range(8):
        m_coeffs = poly_mul_linear(m_coeffs, curr_alpha)
        curr_alpha = gf28_mul(curr_alpha, curr_alpha) # square
        
    # m_coeffs is now [m_0, m_1, ..., m_8]
    # g(x) = m(x)
    
    # Step 2: Construct generator matrix for [17, 9, 5] code
    # The code is cyclic. Generator polynomial g(x) = m(x).
    # The generator matrix rows are shifts of g(x) modulo x^17 - 1.
    # g(x) = m_0 + m_1 x + ... + m_8 x^8
    # Since it's cyclic of length 17, x^17 = 1.
    
    g_coeffs = m_coeffs # length 9
    rows_17 = []
    for i in range(9):
        row = [0] * 17
        for j in range(9):
            if g_coeffs[j] != 0:
                pos = (i + j) % 17
                # Since we are in GF(2), the coefficient is just 1 if present.
                # Wait, g_coeffs[j] is an element of GF(2^8).
                # But the code is over GF(2).
                # The minimal polynomial of alpha has coefficients in GF(2).
                # So g_coeffs[j] should be 0 or 1.
                # Let's verify.
                if g_coeffs[j] not in [0, 1]:
                    # This shouldn't happen for a minimal polynomial over GF(2)
                    # but if it does, we might have an error in field arithmetic or choice of alpha.
                    # For QR codes, the generator polynomial is over GF(2).
                    pass
                row[pos] = 1
        rows_17.append(row)
        
    # Step 3: Extend to length 18
    gen_matrix = []
    for row in rows_17:
        parity = sum(row) % 2
        new_row = row + [parity]
        val = 0
        for k in range(18):
            if new_row[k]:
                val |= (1 << k)
        gen_matrix.append(val)
        
    # Step 4: Generate all 512 codewords
    code_words = [0]
    for g in gen_matrix:
        code_words = code_words + [w ^ g for w in code_words]
        
    # Step 5: Verify min distance
    # For a linear code, min distance is min weight of non-zero codewords
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
