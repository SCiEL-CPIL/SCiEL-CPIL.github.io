import time
import numpy as np

DIMENSIONS = [4, 4, 9]
DEADLINE = 135.0

# Strassen 2x2x2 decomposition
STRASSEN_TERMS = [
    {'a': [1, 0, 0, 1], 'b': [1, 0, 0, 1]},
    {'a': [0, 0, 0, 1], 'b': [-1, 0, 1, 0]},
    {'a': [0, 0, 1, 1], 'b': [0, 1, 0, 0]},
    {'a': [1, 0, 0, 0], 'b': [0, 1, 0, -1]},
    {'a': [1, 1, 0, 0], 'b': [1, 0, 0, 1]},
    {'a': [0, 1, 0, -1], 'b': [0, 0, 1, 1]},
    {'a': [-1, 0, 1, 0], 'b': [1, 1, 0, 0]},
]

C_BLOCKS = [
    [1, 1, -1, 0, 0, 0, 1],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 0],
    [1, -1, 1, 0, 0, -1, 0],
]


def generate_4x4x4_strassen_terms():
    """Generate terms for 4x4 * 4x4 -> 4x4 using Strassen-of-Strassen."""
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


def generate_4x4x5_strassen_schoolbook_terms():
    """Generate terms for 4x4 * 4x5 -> 4x5.
    Split B into 4x4 + 4x1.
    Use Strassen for 4x4*4x4 part, schoolbook for 4x4*4x1 part.
    Returns list of (a_global, b_global, c_global, coeff) where
    a_global = i*4+j, b_global = j*5+k, c_global = k*4+i
    """
    terms = []
    
    # Part 1: 4x4 * 4x4 (columns 0-3 of B)
    terms_4x4 = generate_4x4x4_strassen_terms()
    for (a, b, c, coeff) in terms_4x4:
        # a = i*4 + j
        # b_old = j*4 + k_old (k_old in 0..3)
        # c_old = k_old*4 + i
        j = b // 4
        k_old = b % 4
        b_new = j * 5 + k_old
        # c remains k_old*4 + i, which is correct for 4x5 output
        terms.append((a, b_new, c, coeff))
    
    # Part 2: 4x4 * 4x1 (column 4 of B) - schoolbook
    for i in range(4):
        for j in range(4):
            k = 4
            a_global = i * 4 + j
            b_global = j * 5 + k
            c_global = k * 4 + i
            terms.append((a_global, b_global, c_global, 1))
    
    return terms


def generate_4x4x9_hybrid_terms():
    """Generate terms for 4x4 * 4x9 -> 4x9.
    Split B into 4x4 + 4x5.
    Use Strassen for 4x4*4x4, hybrid for 4x4*4x5.
    """
    terms = []
    
    # Part 1: 4x4 * 4x4 (columns 0-3 of B)
    terms_4x4 = generate_4x4x4_strassen_terms()
    for (a, b, c, coeff) in terms_4x4:
        j = b // 4
        k_old = b % 4
        b_new = j * 9 + k_old
        terms.append((a, b_new, c, coeff))
    
    # Part 2: 4x4 * 4x5 (columns 4-8 of B)
    terms_4x5 = generate_4x4x5_strassen_schoolbook_terms()
    for (a, b, c, coeff) in terms_4x5:
        # b_old = j*5 + k_old (k_old in 0..4)
        j = b // 5
        k_old = b % 5
        b_new = j * 9 + (k_old + 4)  # offset by 4
        # c_old = k_old*4 + i, but k_old here is 0..4, and we want k_new = k_old + 4
        # c_new = (k_old+4)*4 + i
        k2 = c // 4
        i2 = c % 4
        c_new = (k_old + 4) * 4 + i2
        terms.append((a, b_new, c_new, coeff))
    
    return terms


def validate_int(U, V, W, n, m, p, r, scale=1):
    """Validate using integer arithmetic."""
    s = scale
    s3 = s * s * s
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


def schoolbook_decomposition(n, m, p):
    """Generate schoolbook decomposition."""
    r = n * m * p
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return U, V, W


def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    
    def check_time():
        return time.time() - t0 < DEADLINE
    
    # Try hybrid approach: Strassen for 4x4 part, schoolbook for rest
    all_terms = generate_4x4x9_hybrid_terms()
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
        U, V, W = schoolbook_decomposition(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
