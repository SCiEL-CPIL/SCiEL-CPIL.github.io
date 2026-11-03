import time

DIMENSIONS = [4, 4, 9]
DEADLINE = 135.0  # safety margin before 150s kill

# Strassen 2x2x2 decomposition
# M0 = (A00+A11)*(B00+B11)
# M1 = A11*(B10-B00)
# M2 = (A10+A11)*B01
# M3 = A00*(B01-B11)
# M4 = (A00+A01)*(B11+B00)
# M5 = (A01-A11)*(B10+B11)
# M6 = (A10-A00)*(B00+B01)
#
# C00 = M0 + M3 - M4 + M6
# C01 = M2 + M4
# C10 = M1 + M3
# C11 = M0 - M1 + M2 - M5

STRASSEN_A = [
    # M0: (A00+A11)
    [1, 0, 1, 0],  # A00=1, A01=0, A10=1, A11=0 -> wait, (A00+A11) means A00 and A11
    # Let me use 2x2 block indexing: A = [[A00,A01],[A10,A11]]
    # M0: L=A00+A11, R=B00+B11
    # In 4-element form (row-major): A00, A01, A10, A11
    [1, 0, 0, 1],  # M0: A00 + A11
    [0, 0, 1, 1],  # M1: A11 -> wait, M1 = A11*(B10-B00)
    # Let me redo this carefully
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # M0 = (a00+a11)(b00+b11)
    # M1 = a11(b10-b00)
    # M2 = (a10+a11)b01
    # M3 = a00(b01-b11)
    # M4 = (a00+a01)(b11+b00)
    # M5 = (a01-a11)(b10+b11)
    # M6 = (a10-a00)(b00+b01)
]

# Let me define the Strassen terms properly
# For each M, the A-coefficients and B-coefficients in [a00,a01,a10,a11] and [b00,b01,b10,b11] order
STRASSEN_TERMS = [
    # M0 = (a00+a11)(b00+b11)
    {'a': [1, 0, 0, 1], 'b': [1, 0, 0, 1]},
    # M1 = a11(b10-b00)
    {'a': [0, 0, 0, 1], 'b': [-1, 0, 1, 0]},
    # M2 = (a10+a11)b01
    {'a': [0, 0, 1, 1], 'b': [0, 1, 0, 0]},
    # M3 = a00(b01-b11)
    {'a': [1, 0, 0, 0], 'b': [0, 1, 0, -1]},
    # M4 = (a00+a01)(b11+b00)
    {'a': [1, 1, 0, 0], 'b': [1, 0, 0, 1]},
    # M5 = (a01-a11)(b10+b11)
    {'a': [0, 1, 0, -1], 'b': [0, 0, 1, 1]},
    # M6 = (a10-a00)(b00+b01)
    {'a': [-1, 0, 1, 0], 'b': [1, 1, 0, 0]},
]

# C block contributions: C = [[C00, C01], [C10, C11]]
# C00 = M0 + M3 - M4 + M6
# C01 = M2 + M4
# C10 = M1 + M3
# C11 = M0 - M1 + M2 - M5
C_BLOCKS = [
    [1, 1, -1, 0, 0, 0, 1],   # C00: M0, M3, -M4, M6
    [0, 0, 1, 0, 1, 0, 0],   # C01: M2, M4
    [0, 1, 0, 1, 0, 0, 0],   # C10: M1, M3
    [1, -1, 1, 0, 0, -1, 0], # C11: M0, -M1, M2, -M5
]


def strassen_2x2x2_terms():
    """Return list of (a_idx, b_idx, c_idx, coeff) for 2x2x2 Strassen.
    a_idx, b_idx, c_idx are in 0..3 (row-major 2x2 indices)."""
    terms = []
    for mi in range(7):
        a_coeff = STRASSEN_TERMS[mi]['a']
        b_coeff = STRASSEN_TERMS[mi]['b']
        for cbi in range(4):
            # cbi is the output block index (0=C00, 1=C01, 2=C10, 3=C11)
            sign = C_BLOCKS[cbi][mi]
            if sign == 0:
                continue
            for ai in range(4):
                if a_coeff[ai] == 0:
                    continue
                for bi in range(4):
                    if b_coeff[bi] == 0:
                        continue
                coeff = sign * a_coeff[ai] * b_coeff[bi]
                # c_idx in the 2x2 output: need to map cbi to (ci, cj)
                ci, cj = divmod(cbi, 2)
                c_idx = ci * 2 + cj
                terms.append((ai, bi, c_idx, coeff))
    return terms


def strassen_4x4x4_terms():
    """Return list of (a_global, b_global, c_global, coeff) for 4x4x4 Strassen-of-Strassen.
    Global indices: a_global = ai*4 + aj (row ai, col aj of A 4x4)
                   b_global = bj*4 + bk (row bj, col bk of B 4x4)  
                   c_global = ci*4 + ck (row ci, col ck of C 4x4)
    Wait, B is 4x9 here, but for the 4x4 block we treat it as 4x4.
    Actually let me think about this differently.
    
    For 4x4 * 4x4 -> 4x4:
    Split A into 2x2 blocks of 2x2: A = [[A00,A01],[A10,A11]] where each is 2x2
    Split B similarly: B = [[B00,B01],[B10,B11]]
    Output C = [[C00,C01],[C10,C11]]
    
    Outer Strassen gives 7 intermediate 2x2 matrix products.
    Each intermediate is a 2x2 * 2x2 -> 2x2 product, which we do via inner Strassen (7 terms each).
    Total: 7 * 7 = 49 scalar terms.
    
    But each inner term is a scalar product a_ij * b_jk contributing to c_ik.
    So total 49 terms, each with a scalar coefficient.
    """
    # Outer Strassen: 7 terms, each is product of two 2x2 matrices
    # For each outer term s, we have L_s (2x2) and R_s (2x2)
    # L_s[ai][aj] = sum over (bi,bj) in block: coeff * A[block_r][block_c]
    # Actually, let me just enumerate all scalar terms directly.
    
    # A 4x4: A[i][j] where i,j in 0..3
    # Block structure: A = [[A00, A01], [A10, A11]]
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    
    # For outer term s, L_s is a 2x2 matrix where:
    # L_s[0][0] = coeff * A[br0*2+0][bc0*2+0] for each contributing A block
    # This is getting complex. Let me just hardcode by generating all 49 terms.
    
    # Alternative: just use the 2x2x2 Strassen recursively.
    # Outer level: 7 Strassen terms on 2x2 blocks
    # Inner level: each 2x2 block multiplication uses 7 Strassen terms
    
    terms = []
    for outer_s in range(7):
        a_coeff = STRASSEN_TERMS[outer_s]['a']
        b_coeff = STRASSEN_TERMS[outer_s]['b']
        # a_coeff and b_coeff are in [block00, block01, block10, block11] order for 2x2 blocks
        # For each output block position (cbi in 0..3), get the sign
        for cbi in range(4):
            outer_sign = C_BLOCKS[cbi][outer_s]
            if outer_sign == 0:
                continue
            # cbi maps to output block: cbi=0 -> C[0:2,0:2], cbi=1 -> C[0:2,2:4], etc.
            ci_block, cj_block = divmod(cbi, 2)
            # Now, for this outer term, we're computing L_s * R_s where L_s, R_s are 2x2
            # and the result contributes to output block (ci_block, cj_block)
            # L_s[li][lj] = sum over (ai,aj) where a_coeff[(ai,aj)] != 0: a_coeff * A[ai*2+li, aj*2+lj]
            # Wait, a_coeff is indexed by [a00,a01,a10,a11] = block indices
            # a00 = A[0:2,0:2], a01 = A[0:2,2:4], a10 = A[2:4,0:2], a11 = A[2:4,2:4]
            # So a_coeff[0] multiplies A[0:2,0:2], a_coeff[1] multiplies A[0:2,2:4], etc.
            
            # For the inner product L_s * R_s -> result 2x2:
            # We use inner Strassen on the 2x2 matrices
            # L_s = [[l00, l01], [l10, l11]]
            # R_s = [[r00, r01], [r10, r11]]
            # Where l_ij = sum over A blocks: a_coeff[block] * A[2*block_r + i, 2*block_c + j]
            # and r_ij = sum over B blocks: b_coeff[block] * B[2*block_r + i, 2*block_c + j]
            
            # Inner Strassen terms for 2x2*2x2:
            for inner_s in range(7):
                ia_coeff = STRASSEN_TERMS[inner_s]['a']
                ib_coeff = STRASSEN_TERMS[inner_s]['b']
                # ia_coeff = [l00, l01, l10, l11] coefficients
                # ib_coeff = [r00, r01, r10, r11] coefficients
                for inner_cbi in range(4):
                    inner_sign = C_BLOCKS[inner_cbi][inner_s]
                    if inner_sign == 0:
                        continue
                    # inner_cbi maps to position within the 2x2 output block
                    inner_ci, inner_cj = divmod(inner_cbi, 2)
                    # Global output: row = ci_block*2 + inner_ci, col = cj_block*2 + inner_cj
                    c_row = ci_block * 2 + inner_ci
                    c_col = cj_block * 2 + inner_cj
                    c_global = c_row * 4 + c_col  # row-major in 4x4
                    
                    # Now, for each (li, lj) where ia_coeff != 0 and (ri, rj) where ib_coeff != 0:
                    for li in range(2):
                        for lj in range(2):
                            if ia_coeff[li*2+lj] == 0:
                                continue
                            for ri in range(2):
                                for rj in range(2):
                                    if ib_coeff[ri*2+rj] == 0:
                                        continue
                                    # l[li][lj] = sum over A blocks: a_coeff[ab] * A[2*abr + li, 2*abc + lj]
                                    # r[ri][rj] = sum over B blocks: b_coeff[bb] * B[2*abr + ri, 2*abc + rj]
                                    # Product contributes: ia_coeff[li*2+lj] * ib_coeff[ri*2+rj] * inner_sign * outer_sign
                                    # to sum over A blocks and B blocks of A[...] * B[...]
                                    
                                    # Wait, this isn't right. The inner Strassen works on the 2x2 matrices L_s and R_s.
                                    # L_s[li][lj] is a scalar (already summed over A blocks).
                                    # So the inner term is: L_s[li][lj] * R_s[ri][rj] contributing to the 2x2 output.
                                    # But L_s[li][lj] = sum_ab a_coeff[ab] * A[2*abr+li, 2*abc+lj]
                                    # And R_s[ri][rj] = sum_bb b_coeff[bb] * B[2*abr+ri, 2*abc+rj]
                                    # So the product is a sum over pairs of (ab, bb) of A[...] * B[...]
                                    # Each such pair is a separate scalar term!
                                    
                                    for ab in range(4):
                                        if a_coeff[ab] == 0:
                                            continue
                                        # ab: 0=A00, 1=A01, 2=A10, 3=A11
                                        abr, abc = divmod(ab, 2)
                                        a_row = 2*abr + li
                                        a_col = 2*abc + lj
                                        a_global = a_row * 4 + a_col
                                        
                                        for bb in range(4):
                                            if b_coeff[bb] == 0:
                                                continue
                                            # bb: 0=B00, 1=B01, 2=B10, 3=B11
                                            # For B, the block indices are also [B00,B01,B10,B11]
                                            # But wait, B is 4x4 in this subproblem, so B[2*bb_r + ri, 2*bb_c + rj]
                                            bbr, bbc = divmod(bb, 2)
                                            b_row = 2*bbr + ri
                                            b_col = 2*bbc + rj
                                            b_global = b_row * 4 + b_col
                                            
                                            coeff = (outer_sign * inner_sign * 
                                                   a_coeff[ab] * ia_coeff[li*2+lj] *
                                                   b_coeff[bb] * ib_coeff[ri*2+rj])
                                            if coeff != 0:
                                                terms.append((a_global, b_global, c_global, coeff))
    return terms


def build_decomposition(terms):
    """Build U, V, W from list of (a_global, b_global, c_global, coeff) terms.
    U has shape (n*m, rank) = (16, rank)
    V has shape (m*p, rank) = (36, rank)  
    W has shape (p*n, rank) = (36, rank)
    
    Wait, for 4x4*4x9:
    A is 4x4, B is 4x9, C is 4x9
    U: (n*m, rank) = (16, rank) indexed by (i,j)
    V: (m*p, rank) = (36, rank) indexed by (j,k)
    W: (p*n, rank) = (36, rank) indexed by (k,i)
    
    The tensor T[i,j,k] = 1 for all i in 0..3, j in 0..3, k in 0..8
    Reconstruction: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] should equal 1 for all i,j,k
    
    For each term (a_global, b_global, c_global, coeff):
    a_global = i*4 + j -> i = a_global//4, j = a_global%4
    b_global = j'*4 + k -> but B is 4x9, so b_global = j*9 + k? No wait.
    
    Hmm, I need to be careful. In the standard formulation:
    U is indexed by (i,j) where i is row of A (0..3), j is col of A (0..3)
    V is indexed by (j,k) where j is row of B (0..3), k is col of B (0..8)
    W is indexed by (k,i) where k is col of B (0..8), i is row of C (0..3)
    
    So for B 4x9, b_global should be j*9 + k where j in 0..3, k in 0..8.
    And c_global = k*4 + i where k in 0..8, i in 0..3.
    
    But in my Strassen terms for 4x4x4, I was using b_global = b_row*4 + b_col where both are 0..3.
    That's for B being 4x4. For B being 4x9, I need to handle the 9 columns.
    
    Let me reconsider. I'll handle B as three 4x3 blocks or use a different approach.
    Actually, let me just do: split B into B_left (4x4) and B_right (4x5).
    For B_left, use 4x4x4 Strassen (49 terms).
    For B_right, use schoolbook (4*4*5 = 80 terms).
    Total: 129 terms.
    
    Or split B into B1(4x3), B2(4x3), B3(4x3). Each uses schoolbook: 4*4*3=48 each, total 144.
    That's the same as full schoolbook.
    
    Better: B_left 4x4 with Strassen (49) + B_right 4x5 schoolbook (80) = 129.
    """
    n, m, p = DIMENSIONS
    r = len(terms)
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    
    for c, (a_global, b_global, c_global, coeff) in enumerate(terms):
        # a_global = i*m + j = i*4 + j
        i = a_global // m
        j = a_global % m
        # b_global = j*p + k = j*9 + k
        j2 = b_global // p
        k = b_global % p
        # c_global = k*n + i = k*4 + i
        k2 = c_global // n
        i2 = c_global % n
        
        # Should have j == j2 and i == i2
        U[i*m + j][c] += coeff
        V[j*p + k][c] += 1
        W[k*n + i][c] += 1
    
    return U, V, W


def validate_int(U, V, W, n, m, p, r, scale=1):
    """Validate using integer arithmetic. Assumes all entries are integers or dyadic with common scale."""
    s = scale
    s3 = s * s * s
    # Convert to scaled integers
    Us = [[0]*r for _ in range(n*m)]
    Vs = [[0]*r for _ in range(m*p)]
    Ws = [[0]*r for _ in range(p*n)]
    
    for i in range(n*m):
        for c in range(r):
            v = U[i][c]
            if isinstance(v, float):
                Us[i][c] = int(round(v * s))
            else:
                Us[i][c] = int(v) * s
    for i in range(m*p):
        for c in range(r):
            v = V[i][c]
            if isinstance(v, float):
                Vs[i][c] = int(round(v * s))
            else:
                Vs[i][c] = int(v) * s
    for i in range(p*n):
        for c in range(r):
            v = W[i][c]
            if isinstance(v, float):
                Ws[i][c] = int(round(v * s))
            else:
                Ws[i][c] = int(v) * s
    
    for i in range(n):
        for j in range(m):
            Ui = Us[i*m + j]
            for k in range(p):
                Vk = Vs[j*p + k]
                Wk = Ws[k*n + i]
                tot = 0
                for c in range(r):
                    tot += Ui[c] * Vk[c] * Wk[c]
                if tot != s3:
                    return False
    return True


def generate_4x4x4_strassen_terms():
    """Generate terms for 4x4 * 4x4 -> 4x4 using Strassen-of-Strassen.
    Returns list of (a_global, b_global, c_global, coeff) where
    a_global = i*4+j, b_global = j*4+k, c_global = k*4+i
    (for 4x4 matrices, all indices 0..3)
    """
    terms = []
    for outer_s in range(7):
        a_coeff = STRASSEN_TERMS[outer_s]['a']
        b_coeff = STRASSEN_TERMS[outer_s]['b']
        for cbi in range(4):
            outer_sign = C_BLOCKS[cbi][outer_s]
            if outer_sign == 0:
                continue
            ci_block, cj_block = divmod(cbi, 2)
            for inner_s in range(7):
                ia_coeff = STRASSEN_TERMS[inner_s]['a']
                ib_coeff = STRASSEN_TERMS[inner_s]['b']
                for inner_cbi in range(4):
                    inner_sign = C_BLOCKS[inner_cbi][inner_s]
                    if inner_sign == 0:
                        continue
                    inner_ci, inner_cj = divmod(inner_cbi, 2)
                    c_row = ci_block * 2 + inner_ci
                    c_col = cj_block * 2 + inner_cj
                    c_global = c_row * 4 + c_col
                    for li in range(2):
                        for lj in range(2):
                            ia_c = ia_coeff[li*2+lj]
                            if ia_c == 0:
                                continue
                            for ri in range(2):
                                for rj in range(2):
                                    ib_c = ib_coeff[ri*2+rj]
                                    if ib_c == 0:
                                        continue
                                    for ab in range(4):
                                        ac = a_coeff[ab]
                                        if ac == 0:
                                            continue
                                        abr, abc = divmod(ab, 2)
                                        a_row = 2*abr + li
                                        a_col = 2*abc + lj
                                        a_global = a_row * 4 + a_col
                                        for bb in range(4):
                                            bc = b_coeff[bb]
                                            if bc == 0:
                                                continue
                                            bbr, bbc = divmod(bb, 2)
                                            b_row = 2*bbr + ri
                                            b_col = 2*bbc + rj
                                            b_global = b_row * 4 + b_col
                                            coeff = outer_sign * inner_sign * ac * ia_c * bc * ib_c
                                            if coeff != 0:
                                                terms.append((a_global, b_global, c_global, coeff))
    # Remove duplicate terms by combining
    term_dict = {}
    for t in terms:
        key = (t[0], t[1], t[2])
        if key in term_dict:
            term_dict[key] += t[3]
        else:
            term_dict[key] = t[3]
    result = []
    for (a, b, c), coeff in term_dict.items():
        if coeff != 0:
            result.append((a, b, c, coeff))
    return result


def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    
    def check_time():
        return time.time() - t0 < DEADLINE
    
    # Generate 4x4x4 Strassen terms (for the first 4 columns of B)
    terms_4x4 = generate_4x4x4_strassen_terms()
    # Adjust: for B 4x4 -> 4x4, b_global = j*4+k where j,k in 0..3
    # But for B 4x9, we need b_global = j*9+k
    # So remap: for the 4x4 part, b_global_new = j*9 + k where k is 0..3
    terms_left = []
    for (a, b, c, coeff) in terms_4x4:
        # b = j*4 + k_old, where k_old in 0..3
        j = b // 4
        k_old = b % 4
        b_new = j * 9 + k_old
        c_new = c  # c = k*4+i, but now k should be in 0..8 for W indexing
        # c_global = k*n + i = k*4 + i
        # In the 4x4 case, c = c_row*4 + c_col where c_row=k_old(0..3), c_col=i(0..3)
        # So c = k_old*4 + i
        # For the full problem, W is indexed by (k, i) where k in 0..8, i in 0..3
        # So c_new = k_old*4 + i = c (same)
        terms_left.append((a, b_new, c, coeff))
    
    # For the remaining 5 columns (k=4..8), use schoolbook
    terms_right = []
    for i in range(4):
        for j in range(4):
            for k in range(4, 9):
                a_global = i * 4 + j
                b_global = j * 9 + k
                c_global = k * 4 + i
                terms_right.append((a_global, b_global, c_global, 1))
    
    all_terms = terms_left + terms_right
    r = len(all_terms)
    
    # Build U, V, W
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    
    for c, (a, b, cg, coeff) in enumerate(all_terms):
        i = a // m
        j = a % m
        j2 = b // p
        k = b % p
        k2 = cg // n
        i2 = cg % n
        
        U[i*m + j][c] += coeff
        V[j*p + k][c] += 1
        W[k*n + i][c] += 1
    
    # Validate
    if not validate_int(U, V, W, n, m, p, r, scale=1):
        # Fallback to schoolbook
        r_fb = n * m * p
        U = [[0]*r_fb for _ in range(n*m)]
        V = [[0]*r_fb for _ in range(m*p)]
        W = [[0]*r_fb for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
