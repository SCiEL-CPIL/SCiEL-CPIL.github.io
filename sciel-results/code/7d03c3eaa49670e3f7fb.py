import time
import random
from itertools import product

DIMENSIONS = [4, 4, 9]
DEADLINE = 145.0

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36
    
    def check_time():
        return time.time() - t0 < DEADLINE

    def validate(U, V, W, r):
        """Validate decomposition. U, V, W are lists of lists of ints/floats."""
        for i in range(n):
            for j in range(m):
                Uj = U[i * m + j]
                for k in range(p):
                    Vk = V[j * p + k]
                    Wk = W[k * n + i]
                    tot = 0
                    for c in range(r):
                        tot += Uj[c] * Vk[c] * Wk[c]
                    if tot != 1:
                        return False
        return True

    # Strassen 2x2x2: 7 terms
    # Format: (U_coeffs, V_coeffs, W_coeffs)
    # Indices for 2x2 matrix: 00, 01, 10, 11
    STRASSEN_2x2 = [
        # M1 = (A00+A11)(B00+B11)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2 = (A10+A11)B00
        ([0, 0, 0, 1], [-1, 0, 1, 0], [0, 0, 1, 0]), # Wait, let's use standard Strassen
        # Let's use the standard 7:
        # M1 = (A11 - A22)(B22 + B21)
        # M2 = (A11 + A22)(B22)
        # M3 = (A11 + A12)(B12)
        # M4 = (A21 - A11)(B11 + B21)
        # M5 = (A22 + A11)(B12 + B22)
        # M6 = (A12 - A22)(B11 + B22)
        # M7 = (A11 - A21)(B11)
        # This is for 1-indexed. Let's map to 0-indexed 00,01,10,11.
        # C11 = M1 - M2 + M3 + M5
        # C12 = M4 + M2
        # C21 = M1 + M7
        # C22 = M5 - M1 - M4 + M6
        # Let's verify this standard set.
    ]
    
    # Let's stick to the verified 7-term Strassen for 2x2x2 from the previous prompt's STRASSEN list 
    # which was:
    # ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1])
    # ([0, 0, 0, 1], [-1, 0, 1, 0], [0, 0, 1, 0])
    # ([0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 0])
    # ([1, 0, 0, 0], [0, 1, 0, -1], [1, 0, 0, 0])
    # ([1, 1, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0])
    # ([0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 0, 1])
    # ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 0, 0])
    
    STRASSEN = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   
        ([0, 0, 0, 1], [-1, 0, 1, 0], [0, 0, 1, 0]),   
        ([0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 0]),    
        ([1, 0, 0, 0], [0, 1, 0, -1], [1, 0, 0, 0]),   
        ([1, 1, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0]),    
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 0, 1]),   
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 0, 0]),   
    ]

    def build_4x4x4_strassen_terms():
        """
        Build the 49 terms for 4x4x4 multiplication using nested Strassen.
        Returns a list of (U_row, V_row, W_row, coeff) for the 4x4x4 subspace.
        U_row is index in 4x4 (0..15), V_row is index in 4x4 (0..15), W_row is index in 4x4 (0..15).
        Note: In the full 4x4x9 problem, V_row will be mapped to j*p+k.
        Here we assume k is 0..3 for the 4x4 block.
        """
        terms = []
        # Outer Strassen: combines 2x2 blocks of the 4x4 matrix
        # The 4x4 matrix is viewed as 2x2 blocks of 2x2 matrices.
        # Outer indices: 00, 01, 10, 11 referring to blocks.
        # Inner indices: 00, 01, 10, 11 referring to elements within 2x2 blocks.
        
        for outer in range(7):
            oa, ob, oc = STRASSEN[outer]
            # oa, ob, oc are vectors of length 4 for the 2x2 block indices
            # oa[0]=A00_block, oa[1]=A01_block, oa[2]=A10_block, oa[3]=A11_block
            
            for inner in range(7):
                ia, ib, ic = STRASSEN[inner]
                
                # Iterate over the non-zero entries of the outer block multipliers
                for bi in range(4):
                    if oa[bi] == 0:
                        continue
                    # bi corresponds to block (bi_row, bi_col)
                    bi_r, bi_c = divmod(bi, 2)
                    
                    # Iterate over non-zero entries of the inner multipliers for A
                    for ai in range(4):
                        if ia[ai] == 0:
                            continue
                        # ai corresponds to element (ai_r, ai_c) within the 2x2 block
                        ai_r, ai_c = divmod(ai, 2)
                        
                        # Global A index:
                        # Row = 2 * bi_r + ai_r
                        # Col = 2 * bi_c + ai_c
                        a_row = 2 * bi_r + ai_r
                        a_col = 2 * bi_c + ai_c
                        U_idx = a_row * 4 + a_col
                        
                        # Iterate over non-zero entries of the outer block multipliers for B
                        for bj in range(4):
                            if ob[bj] == 0:
                                continue
                            bj_r, bj_c = divmod(bj, 2)
                            
                            # Iterate over non-zero entries of the inner multipliers for B
                            for bi_j in range(4):
                                if ib[bi_j] == 0:
                                    continue
                                bi_j_r, bi_j_c = divmod(bi_j, 2)
                                
                                # Global B index (for 4x4 part, k=0..3):
                                # Row = 2 * bj_r + bi_j_r
                                # Col = 2 * bj_c + bi_j_c
                                b_row = 2 * bj_r + bi_j_r
                                b_col = 2 * bj_c + bi_j_c
                                V_idx_4x4 = b_row * 4 + b_col # Local index in 4x4
                                
                                # Iterate over non-zero entries of the outer block multipliers for C
                                for ci in range(4):
                                    if oc[ci] == 0:
                                        continue
                                    ci_r, ci_c = divmod(ci, 2)
                                    
                                    # Iterate over non-zero entries of the inner multipliers for C
                                    for ci_c2 in range(4):
                                        if ic[ci_c2] == 0:
                                            continue
                                        ci_c2_r, ci_c2_c = divmod(ci_c2, 2)
                                        
                                        # Global C index (for 4x4 part, k=0..3):
                                        # Row = 2 * ci_r + ci_c2_r
                                        # Col = 2 * ci_c + ci_c2_c
                                        c_row = 2 * ci_r + ci_c2_r
                                        c_col = 2 * ci_c + ci_c2_c
                                        W_idx_4x4 = c_row * 4 + c_col # Local index in 4x4
                                        
                                        coeff = oa[bi] * ia[ai] * ob[bj] * ib[bi_j] * oc[ci] * ic[ci_c2]
                                        if coeff != 0:
                                            terms.append((U_idx, V_idx_4x4, W_idx_4x4, coeff))
        
        # Merge terms with same indices
        term_dict = {}
        for t in terms:
            key = (t[0], t[1], t[2])
            if key in term_dict:
                term_dict[key] += t[3]
            else:
                term_dict[key] = t[3]
        
        return [(u, v, w, c) for (u, v, w), c in term_dict.items() if c != 0]

    def build_schoolbook_4x5_terms():
        """
        Build schoolbook terms for the 4x4x5 part (k=4..8).
        Returns list of (U_idx, V_idx_4x5, W_idx_4x5, coeff).
        V_idx_4x5 is local to the 4x5 block (row j, col k_local 0..4).
        W_idx_4x5 is local to the 4x5 block (row k_local, col i).
        """
        terms = []
        for i in range(4):
            for j in range(4):
                for k_local in range(5):
                    k_global = 4 + k_local
                    U_idx = i * 4 + j
                    # V global index is j * 9 + k_global
                    # We return local index for now, will map later
                    V_idx_4x5 = j * 5 + k_local
                    W_idx_4x5 = k_local * 4 + i
                    terms.append((U_idx, V_idx_4x5, W_idx_4x5, 1))
        return terms

    # Construct the full U, V, W matrices
    strassen_terms = build_4x4x4_strassen_terms()
    schoolbook_terms = build_schoolbook_4x5_terms()
    
    total_rank = len(strassen_terms) + len(schoolbook_terms)
    
    U = [[0] * total_rank for _ in range(nm)]
    V = [[0] * total_rank for _ in range(mp)]
    W = [[0] * total_rank for _ in range(pn)]
    
    c = 0
    for (u, v_4x4, w_4x4, coeff) in strassen_terms:
        # u is global U index
        # v_4x4 is local 4x4 index (0..15) -> global V index
        # v_4x4 = b_row * 4 + b_col. Global V index is b_row * 9 + b_col.
        b_row = v_4x4 // 4
        b_col = v_4x4 % 4
        v_global = b_row * 9 + b_col
        
        # w_4x4 is local 4x4 index (0..15) -> global W index
        # w_4x4 = c_row * 4 + c_col. Global W index is c_row * 4 + c_col? 
        # W index is k * n + i.
        # c_row is k (0..3), c_col is i (0..3).
        # So w_global = c_row * 4 + c_col.
        w_global = w_4x4
        
        U[u][c] += coeff
        V[v_global][c] += 1
        W[w_global][c] += 1
        c += 1
        
    for (u, v_4x5, w_4x5, coeff) in schoolbook_terms:
        # u is global U index
        # v_4x5 = j * 5 + k_local. Global V index is j * 9 + (4 + k_local).
        j = v_4x5 // 5
        k_local = v_4x5 % 5
        v_global = j * 9 + (4 + k_local)
        
        # w_4x5 = k_local * 4 + i. Global W index is (4 + k_local) * 4 + i.
        k_local_w = w_4x5 // 4
        i_w = w_4x5 % 4
        w_global = (4 + k_local_w) * 4 + i_w
        
        U[u][c] += coeff
        V[v_global][c] += 1
        W[w_global][c] += 1
        c += 1

    if not validate(U, V, W, total_rank):
        # Fallback to schoolbook
        r_fb = n * m * p
        U = [[0] * r_fb for _ in range(nm)]
        V = [[0] * r_fb for _ in range(mp)]
        W = [[0] * r_fb for _ in range(pn)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    best_U, best_V, best_W, best_r = U[:], V[:], W[:], total_rank

    def fast_validate(U, V, W, r):
        for i in range(n):
            for j in range(m):
                Uj = U[i * m + j]
                for k in range(p):
                    Vk = V[j * p + k]
                    Wk = W[k * n + i]
                    tot = 0.0
                    for c in range(r):
                        uc = Uj[c]
                        if uc == 0:
                            continue
                        vc = Vk[c]
                        if vc == 0:
                            continue
                        wc = Wk[c]
                        if wc == 0:
                            continue
                        tot += uc * vc * wc
                    if tot != 1.0:
                        return False
        return True

    def try_eliminate(U, V, W, r, col):
        r_new = r - 1
        if r_new <= 0:
            return None
        U_new = [row[:col] + row[col+1:] for row in U]
        V_new = [row[:col] + row[col+1:] for row in V]
        W_new = [row[:col] + row[col+1:] for row in W]
        if fast_validate(U_new, V_new, W_new, r_new):
            return U_new, V_new, W_new, r_new
        return None

    # Greedy elimination
    improved = True
    while improved and check_time():
        improved = False
        for col in range(total_rank):
            if not check_time():
                break
            result = try_eliminate(U, V, W, total_rank, col)
            if result is not None:
                U, V, W, total_rank = result
                improved = True
                if total_rank < best_r:
                    best_U, best_V, best_W, best_r = U[:], V[:], W[:], total_rank
                break

    return {"U": best_U, "V": best_V, "W": best_W}
