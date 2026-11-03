import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 2, 4, 8
    
    # Dimensions of factor matrices
    dim_u = n * m  # 8
    dim_v = m * p  # 32
    dim_w = p * n  # 16
    
    # Rank-56 decomposition: 8 blocks of 2x2 * 2x2, each rank 7.
    # The 2x4 * 4x8 multiplication is decomposed into 8 independent 2x2 * 2x2 multiplications.
    # Specifically:
    # A is 2x4, B is 4x8, C is 2x8.
    # We split the intermediate dimension m=4 into two groups of 2 (j_g in {0, 1}).
    # We split the output dimension p=8 into four groups of 2 (k_g in {0, 1, 2, 3}).
    # For each (j_g, k_g), we have a 2x2 (A block) * 2x2 (B block) -> 2x2 (C block).
    # Total blocks: 2 * 4 = 8.
    # Each block uses the standard Strassen decomposition of rank 7.
    # Total rank = 8 * 7 = 56.

    total_rank = 56
    
    # Initialize factor matrices
    U = np.zeros((dim_u, total_rank))
    V = np.zeros((dim_v, total_rank))
    W = np.zeros((dim_w, total_rank))
    
    # Strassen factors for 2x2 * 2x2
    # Local indices for 2x2 matrices:
    # A: [[A00, A01], [A10, A11]] -> indices 0,1,2,3
    # B: [[B00, B01], [B10, B11]] -> indices 0,1,2,3
    # C: [[C00, C01], [C10, C11]] -> indices 0,1,2,3
    #
    # Strassen 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define the 7 rank-1 terms for the 2x2*2x2 tensor
    # Each term is (u_vector, v_vector, w_vector)
    # u_vector has 4 entries (local A indices)
    # v_vector has 4 entries (local B indices)
    # w_vector has 4 entries (local C indices)
    
    strassen_terms = []
    
    # M1
    u1 = np.array([1, 0, 0, 1])
    v1 = np.array([1, 0, 0, 1])
    w1 = np.array([1, 0, 0, 1])
    strassen_terms.append((u1, v1, w1))
    
    # M2
    u2 = np.array([0, 0, 1, 1])
    v2 = np.array([1, 0, 0, 0])
    w2 = np.array([0, 0, 1, -1]) # C10 += M2, C11 -= M2
    strassen_terms.append((u2, v2, w2))
    
    # M3
    u3 = np.array([1, 0, 0, 0])
    v3 = np.array([0, 1, 0, -1])
    w3 = np.array([0, 1, 0, 1]) # C01 += M3, C11 += M3
    strassen_terms.append((u3, v3, w3))
    
    # M4
    u4 = np.array([0, 0, 0, 1])
    v4 = np.array([0, 0, 1, -1])
    w4 = np.array([1, 0, 1, 0]) # C00 += M4, C10 += M4
    strassen_terms.append((u4, v4, w4))
    
    # M5
    u5 = np.array([1, 1, 0, 0])
    v5 = np.array([0, 0, 0, 1])
    w5 = np.array([-1, 0, 0, 1]) # C00 -= M5, C11 += M5
    strassen_terms.append((u5, v5, w5))
    
    # M6
    u6 = np.array([-1, 0, 1, 0])
    v6 = np.array([1, 1, 0, 0])
    w6 = np.array([0, 0, 0, 1]) # C11 += M6
    strassen_terms.append((u6, v6, w6))
    
    # M7
    u7 = np.array([0, 1, 0, -1])
    v7 = np.array([0, 0, 1, 1])
    w7 = np.array([1, -1, 0, 0]) # C00 += M7, C01 -= M7
    strassen_terms.append((u7, v7, w7))
    
    col = 0
    # Iterate over the 8 blocks
    # j_g: 0, 1 (splits the 4 intermediate dims into two groups of 2)
    # k_g: 0, 1, 2, 3 (splits the 8 output dims into four groups of 2)
    for j_g in range(2):
        for k_g in range(4):
            # Map local 2x2 indices to global indices
            # Local A (2x2) indices:
            # 0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)
            # Global A index: i*m + j
            # i is 0 or 1. j is 2*j_g + local_j (0 or 1).
            # local_j = idx % 2, i = idx // 2
            
            # Local B (2x2) indices:
            # 0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)
            # Global B index: j*p + k
            # j is 2*j_g + local_j. k is 2*k_g + local_k.
            # local_j = idx // 2, local_k = idx % 2
            
            # Local C (2x2) indices:
            # 0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)
            # Global C index: k*n + i
            # k is 2*k_g + local_k. i is local_i.
            # local_i = idx // 2, local_k = idx % 2
            
            for u_l, v_l, w_l in strassen_terms:
                for local_idx in range(4):
                    # U contribution
                    if u_l[local_idx] != 0:
                        i = local_idx // 2
                        local_j = local_idx % 2
                        global_j = 2 * j_g + local_j
                        global_u_idx = i * m + global_j
                        U[global_u_idx, col] += u_l[local_idx]
                    
                    # V contribution
                    if v_l[local_idx] != 0:
                        local_j = local_idx // 2
                        local_k = local_idx % 2
                        global_j = 2 * j_g + local_j
                        global_k = 2 * k_g + local_k
                        global_v_idx = global_j * p + global_k
                        V[global_v_idx, col] += v_l[local_idx]
                    
                    # W contribution
                    if w_l[local_idx] != 0:
                        local_i = local_idx // 2
                        local_k = local_idx % 2
                        global_k = 2 * k_g + local_k
                        global_i = local_i
                        global_w_idx = global_k * n + global_i
                        W[global_w_idx, col] += w_l[local_idx]
                
                col += 1
    
    # Verification (optional, but good for safety)
    # If time permits, verify the decomposition
    if time.time() < deadline - 5.0:
        T_recon = np.zeros((dim_u, dim_v, dim_w))
        for c in range(total_rank):
            T_recon += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        
        T_orig = np.zeros((dim_u, dim_v, dim_w))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_orig[i*m + j, j*p + k, k*n + i] = 1
        
        if not np.allclose(T_recon, T_orig, atol=1e-9):
            # Fallback to schoolbook if verification fails
            rank_school = n * m * p
            U_school = np.zeros((dim_u, rank_school))
            V_school = np.zeros((dim_v, rank_school))
            W_school = np.zeros((dim_w, rank_school))
            col_s = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_school[i*m + j, col_s] = 1
                        V_school[j*p + k, col_s] = 1
                        W_school[k*n + i, col_s] = 1
                        col_s += 1
            return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
