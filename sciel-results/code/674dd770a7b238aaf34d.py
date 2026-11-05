import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 110.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    n, m, p = 4, 4, 6
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_dict, v_dict, w_dict):
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        
        for (i, j), coeff in u_dict.items():
            u_col[i * m + j] += coeff
        for (j, k), coeff in v_dict.items():
            v_col[j * p + k] += coeff
        for (i, k), coeff in w_dict.items():
            w_col[k * n + i] += coeff
            
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
    
    # Strassen 2x2x2 terms: (U_pattern, V_pattern, W_pattern)
    # Each pattern is a dict mapping (row, col) -> coeff
    strassen_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # 2x2x3 terms: Strassen for cols 0-1 (7 terms) + schoolbook for col 2 (4 terms)
    # Total: 11 terms
    # For 2x2x3: A is 2x2, B is 2x3, C is 2x3
    # Strassen part uses B cols 0,1; schoolbook part uses B col 2
    terms_2x2x3 = []
    for u_pat, v_pat, w_pat in strassen_2x2:
        # v_pat has keys (row, col) where col is 0 or 1 (for 2x2x2)
        # In 2x2x3, B has 3 columns, so v_pat keys stay as (row, col) with col in {0,1}
        # w_pat keys stay as (row, col) with col in {0,1}
        terms_2x2x3.append((u_pat, v_pat, w_pat))
    # Schoolbook for col 2
    for i in range(2):
        for j in range(2):
            u_pat = {(i, j): 1}
            v_pat = {(j, 2): 1}
            w_pat = {(i, 2): 1}
            terms_2x2x3.append((u_pat, v_pat, w_pat))
    
    # Kronecker product: 4x4x6 = (2x2x2) ⊗ (2x2x3)
    # A[4x4] = A_outer[2x2] ⊗ A_inner[2x2] where A_outer[i,j] is a 2x2 block
    # B[4x6] = B_outer[2x2] ⊗ B_inner[2x3] where B_outer[j,k] is a 2x3 block  
    # C[4x6] = C_outer[2x2] ⊗ C_inner[2x3] where C_outer[i,k] is a 2x3 block
    #
    # Indexing:
    # A[i,j] where i = 2*oi + ii, j = 2*oj + jj (oi,oi in 0..1, ii,jj in 0..1)
    # B[j,k] where j = 2*oj + jj, k = 2*ok + kk (oj in 0..1, jj in 0..1, ok in 0..1, kk in 0..2)
    # C[i,k] where i = 2*oi + ii, k = 2*ok + kk (oi in 0..1, ii in 0..1, ok in 0..1, kk in 0..2)
    #
    # The outer 2x2x2 uses Strassen on the "outer" indices (oi,oj,ok)
    # The inner 2x2x3 uses the terms_2x2x3 on the "inner" indices (ii,jj,kk)
    
    for outer_u, outer_v, outer_w in strassen_2x2:
        for inner_u, inner_v, inner_w in terms_2x2x3:
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            for (oi, oj), outer_coeff_u in outer_u.items():
                for (ii, jj), inner_coeff_u in inner_u.items():
                    i = 2 * oi + ii
                    j = 2 * oj + jj
                    if 0 <= i < n and 0 <= j < m:
                        u_col[i * m + j] += outer_coeff_u * inner_coeff_u
            
            for (oj, ok), outer_coeff_v in outer_v.items():
                for (jj, kk), inner_coeff_v in inner_v.items():
                    j = 2 * oj + jj
                    k = 2 * ok + kk
                    if 0 <= j < m and 0 <= k < p:
                        v_col[j * p + k] += outer_coeff_v * inner_coeff_v
            
            for (oi, ok), outer_coeff_w in outer_w.items():
                for (ii, kk), inner_coeff_w in inner_w.items():
                    i = 2 * oi + ii
                    k = 2 * ok + kk
                    if 0 <= i < n and 0 <= k < p:
                        w_col[k * n + i] += outer_coeff_w * inner_coeff_w
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
    
    rank = len(U_cols)
    
    U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m*p)]
    W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p*n)]
    
    # Verify
    U_arr = np.array(U_final, dtype=np.float64)
    V_arr = np.array(V_final, dtype=np.float64)
    W_arr = np.array(W_final, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = np.dot(U_arr[u_idx, :], V_arr[v_idx, :] * W_arr[w_idx, :])
                if abs(total - 1.0) > 1e-9:
                    # Fallback to schoolbook if verification fails
                    rank_fb = n * m * p
                    U_fb = [[0] * rank_fb for _ in range(n * m)]
                    V_fb = [[0] * rank_fb for _ in range(m * p)]
                    W_fb = [[0] * rank_fb for _ in range(p * n)]
                    column = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                U_fb[i * m + j][column] = 1
                                V_fb[j * p + k][column] = 1
                                W_fb[k * n + i][column] = 1
                                column += 1
                    return {"U": U_fb, "V": V_fb, "W": W_fb}
                    
    return {"U": U_final, "V": V_final, "W": W_final}
