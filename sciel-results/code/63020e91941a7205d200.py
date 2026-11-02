import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(np.array(u, dtype=float))
        V_cols.append(np.array(v, dtype=float))
        W_cols.append(np.array(w, dtype=float))
    
    def get_u_idx(gi, gj):
        return gi * m + gj
        
    def get_v_idx(gj, gk):
        return gj * p + gk
        
    def get_w_idx(gk, gi):
        return gk * n + gi

    # Define Strassen terms for a 2x2x2 block
    # A block at (ai, aj), B block at (bj, bk)
    # A is 2x2, B is 2x2, C is 2x2
    # Local indices:
    # A: 00, 01, 10, 11
    # B: 00, 01, 10, 11
    # C: 00, 01, 10, 11
    
    def add_strassen_2x2_2(ai, aj, bj, bk):
        # M1 = (a+d)(e+h)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[get_u_idx(ai, aj)] = 1
        u[get_u_idx(ai+1, aj+1)] = 1
        v[get_v_idx(bj, bk)] = 1
        v[get_v_idx(bj+1, bk+1)] = 1
        w[get_w_idx(bk, ai)] = 1
        w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)
        
        # M2 = (c+d)e
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[get_u_idx(ai+1, aj)] = 1
        u[get_u_idx(ai+1, aj+1)] = 1
        v[get_v_idx(bj, bk)] = 1
        w[get_w_idx(bk, ai+1)] = 1
        w[get_w_idx(bk+1, ai+1)] = -1
        add_term(u, v, w)
        
        # M3 = a(f-h)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[get_u_idx(ai, aj)] = 1
        v[get_v_idx(bj, bk+1)] = 1
        v[get_v_idx(bj+1, bk+1)] = -1
        w[get_w_idx(bk+1, ai)] = 1
        w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)
        
        # M4 = d(g-e)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[get_u_idx(ai+1, aj+1)] = 1
        v[get_v_idx(bj+1, bk)] = 1
        v[get_v_idx(bj, bk)] = -1
        w[get_w_idx(bk, ai)] = 1
        w[get_w_idx(bk, ai+1)] = 1
        add_term(u, v, w)
        
        # M5 = (a+b)h
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[get_u_idx(ai, aj)] = 1
        u[get_u_idx(ai, aj+1)] = 1
        v[get_v_idx(bj+1, bk+1)] = 1
        w[get_w_idx(bk, ai)] = -1
        w[get_w_idx(bk+1, ai)] = 1
        add_term(u, v, w)
        
        # M6 = (c-a)(e+f)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[get_u_idx(ai+1, aj)] = 1
        u[get_u_idx(ai, aj)] = -1
        v[get_v_idx(bj, bk)] = 1
        v[get_v_idx(bj, bk+1)] = 1
        w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)
        
        # M7 = (b-d)(g+h)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[get_u_idx(ai, aj+1)] = 1
        u[get_u_idx(ai+1, aj+1)] = -1
        v[get_v_idx(bj+1, bk)] = 1
        v[get_v_idx(bj+1, bk+1)] = 1
        w[get_w_idx(bk, ai)] = 1
        add_term(u, v, w)

    # We need to decompose 4x4 * 4x5.
    # We can view A as 2x2 blocks of 2x2.
    # B as 2x2 blocks of 2x2, plus a 2x1 block?
    # No, B is 4x5.
    # Let's split B into B1 (4x4) and B2 (4x1).
    # C = A * B1 + A * B2.
    # We can use Strassen for A * B1 (which is 4x4 * 4x4) and Naive for A * B2 (4x4 * 4x1).
    # Strassen for 4x4 * 4x4 has rank 7^2 = 49.
    # Naive for 4x4 * 4x1 has rank 4*4*1 = 16.
    # Total rank = 49 + 16 = 65.
    
    # Let's implement the 4x4 * 4x4 part using Kronecker product of Strassen 2x2.
    # A = [[A11, A12], [A21, A22]] where each Ai is 2x2.
    # B1 = [[B11, B12], [B21, B22]] where each Bj is 2x2.
    # C1 = A * B1.
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # We can compute each of these using Strassen 2x2?
    # No, A11*B11 is 2x2 * 2x2.
    # So we can apply Strassen to each of the 4 products.
    # That would be 4 * 7 = 28 terms.
    # But wait, Strassen 4x4 is rank 49. Why 28?
    # Because the terms in C11, C12, C21, C22 are not independent in the same way?
    # Actually, the standard Strassen 4x4 is obtained by applying Strassen to the 2x2 blocks of A and B.
    # Let's trace it.
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # This requires computing 7 products of 2x2 matrices.
    # Each 2x2 * 2x2 product requires 7 multiplications (Strassen) or 4 (Naive).
    # If we use Strassen for the inner 2x2 products, the total rank is 7 * 7 = 49.
    # If we use Naive for the inner 2x2 products, the total rank is 7 * 4 = 28? No, that's not right.
    # The rank of the tensor is the number of scalar multiplications.
    # In the 4x4 Strassen, we perform 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication is a tensor of rank 7 (if using Strassen) or 4 (if using Naive).
    # So the total rank is 7 * 7 = 49.
    
    # Let's implement the 49-term decomposition.
    # We will iterate over the 7 outer Strassen terms and the 7 inner Strassen terms.
    
    # Define the 7 outer terms as pairs of block indices.
    # Outer term 1: (A11+A22)(B11+B22) -> U picks A11, A22. V picks B11, B22.
    # We need to define the U, V, W vectors for each of the 7 outer terms, where each entry is a 2x2 block.
    # Then we expand each 2x2 block entry using the 7 inner Strassen terms.
    
    # Let's define the outer terms' block coefficients.
    # U_outer[r][block_idx] = coeff
    # V_outer[r][block_idx] = coeff
    # W_outer[r][block_idx] = coeff
    
    # Block indices: 0:11, 1:12, 2:21, 3:22
    
    U_outer = [
        {0: 1, 3: 1},   # M1: A11+A22
        {2: 1, 3: 1},   # M2: A21+A22
        {0: 1},         # M3: A11
        {3: 1},         # M4: A22
        {0: 1, 1: 1},   # M5: A11+A12
        {2: 1, 0: -1},  # M6: A21-A11
        {1: 1, 3: -1},  # M7: A12-A22
    ]
    
    V_outer = [
        {0: 1, 3: 1},   # M1: B11+B22
        {0: 1},         # M2: B11
        {1: 1, 3: -1},  # M3: B12-B22
        {2: 1, 0: -1},  # M4: B21-B11
        {3: 1},         # M5: B22
        {0: 1, 1: 1},   # M6: B11+B12
        {2: 1, 3: 1},   # M7: B21+B22
    ]
    
    W_outer = [
        {0: 1, 3: 1},   # M1: C11, C22
        {2: 1, 3: -1},  # M2: C21, C22
        {1: 1, 3: 1},   # M3: C12, C22
        {0: 1, 2: 1},   # M4: C11, C21
        {0: -1, 1: 1},  # M5: C11, C12
        {3: 1},         # M6: C22
        {0: 1},         # M7: C11
    ]
    
    # Now, for each outer term r (0..6), we have block coefficients.
    # We need to compute the product of the U_block and V_block for each pair of blocks (ub, vb).
    # The product is a 2x2 matrix. We decompose this 2x2 matrix multiplication using the 7 inner Strassen terms.
    # For each inner term s (0..6), we add a rank-1 term.
    
    # Inner Strassen terms for 2x2 * 2x2.
    # We need the U, V, W vectors for the 7 inner terms.
    # These are the same as the `add_strassen_2x2_2` function, but we need the coefficients.
    # Let's define the inner terms' local coefficients.
    # Local indices: 0:00, 1:01, 2:10, 3:11
    
    U_inner = [
        {0: 1, 3: 1},   # M1: a+d
        {2: 1, 3: 1},   # M2: c+d
        {0: 1},         # M3: a
        {3: 1},         # M4: d
        {0: 1, 1: 1},   # M5: a+b
        {2: 1, 0: -1},  # M6: c-a
        {1: 1, 3: -1},  # M7: b-d
    ]
    
    V_inner = [
        {0: 1, 3: 1},   # M1: e+h
        {0: 1},         # M2: e
        {1: 1, 3: -1},  # M3: f-h
        {2: 1, 0: -1},  # M4: g-e
        {3: 1},         # M5: h
        {0: 1, 1: 1},   # M6: e+f
        {2: 1, 3: 1},   # M7: g+h
    ]
    
    W_inner = [
        {0: 1, 3: 1},   # M1: c11, c22
        {2: 1, 3: -1},  # M2: c21, c22
        {1: 1, 3: 1},   # M3: c12, c22
        {0: 1, 2: 1},   # M4: c11, c21
        {0: -1, 1: 1},  # M5: c11, c12
        {3: 1},         # M6: c22
        {0: 1},         # M7: c11
    ]
    
    # We iterate over outer r, inner s.
    # For each (r, s), we construct the U, V, W vectors for the 4x4x4 part.
    
    for r in range(7):
        for s in range(7):
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            
            # U vector:
            # For each block index ub in U_outer[r], and local index li in U_inner[s]:
            # u[global_idx] += U_outer[r][ub] * U_inner[s][li]
            
            for ub, coeff_u in U_outer[r].items():
                # Block ub corresponds to A[bi, bj] where bi, bj are block row/col.
                # ub=0 -> (0,0), ub=1 -> (0,1), ub=2 -> (1,0), ub=3 -> (1,1)
                bi = ub // 2
                bj = ub % 2
                for li, coeff_li in U_inner[s].items():
                    # Local index li corresponds to A[bi*2 + li_row, bj*2 + li_col]
                    li_row = li // 2
                    li_col = li % 2
                    gi = bi * 2 + li_row
                    gj = bj * 2 + li_col
                    u[get_u_idx(gi, gj)] += coeff_u * coeff_li
            
            # V vector:
            for vb, coeff_v in V_outer[r].items():
                vb_row = vb // 2
                vb_col = vb % 2
                for li, coeff_li in V_inner[s].items():
                    li_row = li // 2
                    li_col = li % 2
                    gj = vb_row * 2 + li_row
                    gk = vb_col * 2 + li_col
                    v[get_v_idx(gj, gk)] += coeff_v * coeff_li
            
            # W vector:
            for wb, coeff_w in W_outer[r].items():
                wb_row = wb // 2
                wb_col = wb % 2
                for li, coeff_li in W_inner[s].items():
                    li_row = li // 2
                    li_col = li % 2
                    gk = wb_row * 2 + li_row
                    gi = wb_col * 2 + li_col
                    w[get_w_idx(gk, gi)] += coeff_w * coeff_li
            
            add_term(u, v, w)
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Now add the Naive part for A * B2 (4x4 * 4x1)
    # B2 is the last column of B, indices (0,4), (1,4), (2,4), (3,4).
    # C2 is the last column of C, indices (0,4), (1,4), (2,4), (3,4).
    # C2[i, 4] = sum_j A[i, j] * B[j, 4]
    # We add 16 terms.
    
    if time.time() < deadline:
        for i in range(4):
            for j in range(4):
                u = np.zeros(n*m)
                v = np.zeros(m*p)
                w = np.zeros(p*n)
                u[get_u_idx(i, j)] = 1
                v[get_v_idx(j, 4)] = 1
                w[get_w_idx(4, i)] = 1
                add_term(u, v, w)
    
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    
    # Verify
    if _verify_decomposition(U_mat, V_mat, W_mat, n, m, p):
        return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
    else:
        # Fallback
        U_naive = np.zeros((n*m, n*m*p))
        V_naive = np.zeros((m*p, n*m*p))
        W_naive = np.zeros((p*n, n*m*p))
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_naive[i*m+j, idx] = 1
                    V_naive[j*p+k, idx] = 1
                    W_naive[k*n+i, idx] = 1
                    idx += 1
        
        return {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True
# EVOLVE-BLOCK-END
