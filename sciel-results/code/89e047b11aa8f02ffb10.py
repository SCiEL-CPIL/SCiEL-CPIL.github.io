import time
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 9]
DEADLINE = 135.0

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36
    
    def check_time():
        return time.time() - t0 < DEADLINE
    
    def validate(U, V, W, r):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tot = 0
                    for c in range(r):
                        tot += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    if tot != 1:
                        return False
        return True
    
    # Start with known 129-rank solution
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
        term_dict = {}
        for t in terms:
            key = (t[0], t[1], t[2])
            if key in term_dict:
                term_dict[key] += t[3]
            else:
                term_dict[key] = t[3]
        return [(a, b, c, coeff) for (a, b, c), coeff in term_dict.items() if coeff != 0]
    
    terms_4x4 = generate_4x4x4_strassen_terms()
    terms_left = []
    for (a, b, c, coeff) in terms_4x4:
        j = b // 4
        k_old = b % 4
        b_new = j * 9 + k_old
        terms_left.append((a, b_new, c, coeff))
    
    terms_right = []
    for i in range(4):
        for j in range(4):
            for k in range(4, 9):
                a_global = i * 4 + j
                b_global = j * 9 + k
                c_global = k * 4 + i
                terms_right.append((a_global, b_global, c_global, 1))
    
    all_terms = terms_left + terms_right
    r_best = len(all_terms)
    
    U_best = [[0]*r_best for _ in range(nm)]
    V_best = [[0]*r_best for _ in range(mp)]
    W_best = [[0]*r_best for _ in range(pn)]
    
    for c, (a, b, cg, coeff) in enumerate(all_terms):
        i = a // m
        j = a % m
        j2 = b // p
        k = b % p
        k2 = cg // n
        i2 = cg % n
        U_best[i*m + j][c] += coeff
        V_best[j*p + k][c] += 1
        W_best[k*n + i][c] += 1
    
    if not validate(U_best, V_best, W_best, r_best):
        r_fb = n * m * p
        U = [[0]*r_fb for _ in range(nm)]
        V = [[0]*r_fb for _ in range(mp)]
        W = [[0]*r_fb for _ in range(pn)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to improve via random search
    # This is a structural change: random search instead of constructive
    random.seed(42)
    
    def random_decomposition(r, scale=2):
        """Generate random U, V, W with entries in {-1, 0, 1} * scale."""
        U = [[0]*r for _ in range(nm)]
        V = [[0]*r for _ in range(mp)]
        W = [[0]*r for _ in range(pn)]
        for i in range(nm):
            for c in range(r):
                if random.random() < 0.7:
                    U[i][c] = random.choice([-1, 1])
        for i in range(mp):
            for c in range(r):
                if random.random() < 0.7:
                    V[i][c] = random.choice([-1, 1])
        for i in range(pn):
            for c in range(r):
                if random.random() < 0.7:
                    W[i][c] = random.choice([-1, 1])
        return U, V, W
    
    # Try ranks from r_best-1 down to some lower bound
    # Lower bound: at least max(nm, mp, pn) = 36, and at least the rank of the tensor
    # The tensor rank is at least max(n*m, m*p, p*n) / min(n*m, m*p, p*n) ... not sure
    # Just try a few ranks
    
    if check_time():
        for r_try in range(r_best - 1, max(36, r_best - 20), -1):
            if not check_time():
                break
            attempts = 0
            max_attempts = 50
            while attempts < max_attempts and check_time():
                attempts += 1
                U_try, V_try, W_try = random_decomposition(r_try)
                if validate(U_try, V_try, W_try, r_try):
                    r_best = r_try
                    U_best = U_try
                    V_best = V_try
                    W_best = W_try
                    break
    
    return {"U": U_best, "V": V_best, "W": W_best}
