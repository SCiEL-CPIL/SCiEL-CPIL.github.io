import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    best_rank = 72
    best_U = None
    best_V = None
    best_W = None
    
    def verify(U, V, W, rank):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = 1.0
        
        recon = np.einsum('ic,jc,kc->ijk', U_arr, V_arr, W_arr)
        
        return np.allclose(recon, target, atol=1e-6)

    def add_strassen(i_base, j_base, k_base, U, V, W, col):
        a_idx = i_base * m + j_base
        b_idx = i_base * m + (j_base + 1)
        c_idx = (i_base + 1) * m + j_base
        d_idx = (i_base + 1) * m + (j_base + 1)
        
        e_idx = j_base * p + k_base
        f_idx = j_base * p + (k_base + 1)
        g_idx = (j_base + 1) * p + k_base
        h_idx = (j_base + 1) * p + (k_base + 1)
        
        p_idx = k_base * n + i_base
        q_idx = (k_base + 1) * n + i_base
        r_idx = k_base * n + (i_base + 1)
        s_idx = (k_base + 1) * n + (i_base + 1)
        
        # m1
        U[a_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        # m2
        U[c_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1
        W[r_idx][col] = 1; W[s_idx][col] = -1
        col += 1
        # m3
        U[a_idx][col] = 1
        V[f_idx][col] = 1; V[h_idx][col] = -1
        W[q_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        # m4
        U[d_idx][col] = 1
        V[g_idx][col] = 1; V[e_idx][col] = -1
        W[p_idx][col] = 1; W[r_idx][col] = 1
        col += 1
        # m5
        U[a_idx][col] = 1; U[b_idx][col] = 1
        V[h_idx][col] = 1
        W[p_idx][col] = -1; W[q_idx][col] = 1
        col += 1
        # m6
        U[c_idx][col] = 1; U[a_idx][col] = -1
        V[e_idx][col] = 1; V[f_idx][col] = 1
        W[s_idx][col] = 1
        col += 1
        # m7
        U[b_idx][col] = 1; U[d_idx][col] = -1
        V[g_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1
        col += 1
        return col

    def add_schoolbook(i, j, k, U, V, W, col):
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        return col + 1

    # Strategy 1: Hybrid (schoolbook row 0 + Strassen rows 1-2) = rank 66
    rank1 = 66
    U1 = [[0] * rank1 for _ in range(n * m)]
    V1 = [[0] * rank1 for _ in range(m * p)]
    W1 = [[0] * rank1 for _ in range(p * n)]
    col1 = 0
    # Schoolbook for row 0: 4*6 = 24 terms
    for j in range(m):
        for k in range(p):
            col1 = add_schoolbook(0, j, k, U1, V1, W1, col1)
    # Strassen 2x2x2 blocks for rows 1-2: 2 j-blocks * 3 k-blocks * 7 = 42 terms
    for k_block in range(p // 2):
        for j_block in range(m // 2):
            col1 = add_strassen(1, j_block * 2, k_block * 2, U1, V1, W1, col1)
    
    assert col1 == rank1, f"Expected {rank1} terms, got {col1}"
    if verify(U1, V1, W1, rank1):
        best_rank = rank1
        best_U, best_V, best_W = U1, V1, W1

    if best_U is None:
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col_fb = add_schoolbook(i, j, k, U_fb, V_fb, W_fb, col_fb)
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": best_U, "V": best_V, "W": best_W}
