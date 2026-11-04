import time
import random
import numpy as np

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
        """Vectorized validation using NumPy."""
        Ua = np.array(U, dtype=np.int64)
        Va = np.array(V, dtype=np.int64)
        Wa = np.array(W, dtype=np.int64)
        T = np.zeros((n, m, p), dtype=np.int64)
        for c in range(r):
            ur = Ua[:, c].reshape(n, m)
            vr = Va[:, c].reshape(m, p)
            wr = Wa[:, c].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', ur, vr, wr)
        return np.all(T == 1)
    
    # 3x3 Strassen decomposition (rank 7)
    # For multiplying 3x3 * 3x3, use 3x3 Strassen with rank 7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # But we need 4x4 * 4x9. Let's think differently.
    # The 9 columns can be viewed as 3 blocks of 3 columns each.
    # For each block of 3 columns, we can use a 3x3 Strassen-like approach.
    # Actually, let's use the block structure: 4x4 * 4x9 = (4x4 * 4x3) for 3 blocks
    # But that doesn't directly help with Strassen.
    
    # Better approach: Use the 4x4x4 Strassen for the first 4 columns,
    # and for columns 4-8, use a different decomposition.
    
    # Actually, let's try: treat the 4x4*4x9 as follows:
    # Split the 9 columns into three groups of 3: C1 (cols 0-2), C2 (cols 3-5), C3 (cols 6-8)
    # For each group, we need to compute A * B_i where B_i is 4x3.
    # This is like 4x4 * 4x3 multiplication.
    
    # The rank of 4x4*4x3 is at most 4*3 = 12 (schoolbook), but can be lower.
    # Total rank would be 3 * rank(4x4*4x3).
    
    # Let's just use the existing approach but try to optimize it.
    
    # Start with the existing 4x4x4 Strassen terms
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
    
    # Convert to numpy for faster operations
    U_np = np.array(U_best, dtype=np.int64)
    V_np = np.array(V_best, dtype=np.int64)
    W_np = np.array(W_best, dtype=np.int64)
    
    # Try to reduce rank by finding linear dependencies
    def compute_tensor(Ua, Va, Wa, r):
        """Compute the full tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]"""
        T = np.zeros((n, m, p), dtype=np.int64)
        for c in range(r):
            ur = Ua[:, c].reshape(n, m)
            vr = Va[:, c].reshape(m, p)
            wr = Wa[:, c].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', ur, vr, wr)
        return T
    
    def try_remove_column(Ua, Va, Wa, r, col):
        """Try to remove column col and see if the tensor is still valid."""
        if r <= 1:
            return None
        U_new = np.delete(Ua, col, axis=1)
        V_new = np.delete(Va, col, axis=1)
        W_new = np.delete(Wa, col, axis=1)
        T = compute_tensor(U_new, V_new, W_new, r - 1)
        if np.all(T == 1):
            return U_new, V_new, W_new, r - 1
        return None
    
    def try_merge_columns(Ua, Va, Wa, r, col1, col2):
        """Try to merge col2 into col1 by finding if their tensors are proportional."""
        T1 = np.einsum('i,j,k->ijk', 
                       Ua[:,col1].reshape(n,m), 
                       Va[:,col1].reshape(m,p), 
                       Wa[:,col1].reshape(p,n))
        T2 = np.einsum('i,j,k->ijk', 
                       Ua[:,col2].reshape(n,m), 
                       Va[:,col2].reshape(m,p), 
                       Wa[:,col2].reshape(p,n))
        
        # Check if T2 = alpha * T1 for some alpha
        nonzero = T1 != 0
        if not np.any(nonzero):
            return None
        alphas = T2[nonzero].astype(np.float64) / T1[nonzero].astype(np.float64)
        alpha = alphas[0]
        if not np.allclose(alphas, alpha, atol=1e-9):
            return None
        if not np.allclose(T2, alpha * T1, atol=1e-9):
            return None
        
        # Check if alpha is close to an integer
        alpha_rounded = round(alpha)
        if abs(alpha - alpha_rounded) < 1e-9:
            Ua[:,col1] += alpha_rounded * Ua[:,col2]
            Va[:,col1] += alpha_rounded * Va[:,col2]
            Wa[:,col1] += alpha_rounded * Wa[:,col2]
            U_new = np.delete(Ua, col2, axis=1)
            V_new = np.delete(Va, col2, axis=1)
            W_new = np.delete(Wa, col2, axis=1)
            T = compute_tensor(U_new, V_new, W_new, r - 1)
            if np.all(T == 1):
                return U_new, V_new, W_new, r - 1
        return None
    
    # Local search to reduce rank
    improved = True
    while improved and check_time():
        improved = False
        r_current = U_np.shape[1]
        
        # Try removing individual columns
        for col in range(r_current):
            if not check_time():
                break
            result = try_remove_column(U_np, V_np, W_np, r_current, col)
            if result is not None:
                U_np, V_np, W_np, r_new = result
                improved = True
                break
        
        # Try merging columns
        if not improved and check_time():
            for col1 in range(r_current):
                if not check_time():
                    break
                for col2 in range(col1 + 1, r_current):
                    if not check_time():
                        break
                    U_copy = U_np.copy()
                    V_copy = V_np.copy()
                    W_copy = W_np.copy()
                    result = try_merge_columns(U_copy, V_copy, W_copy, r_current, col1, col2)
                    if result is not None:
                        U_np, V_np, W_np, r_new = result
                        improved = True
                        break
                if improved:
                    break
    
    # Convert back to lists
    U_best = U_np.tolist()
    V_best = V_np.tolist()
    W_best = W_np.tolist()
    
    return {"U": U_best, "V": V_best, "W": W_best}
