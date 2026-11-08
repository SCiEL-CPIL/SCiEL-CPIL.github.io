import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Try rank 55 decomposition
    # 23 terms for 3x3x3, 9 terms for 3x3x1
    rank = 55
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_rank1_term(U_vec, V_vec, W_vec):
        nonlocal col
        if col >= rank:
            return
        for i, val in enumerate(U_vec):
            if val != 0:
                U[i][col] += val
        for i, val in enumerate(V_vec):
            if val != 0:
                V[i][col] += val
        for i, val in enumerate(W_vec):
            if val != 0:
                W[i][col] += val
        col += 1

    def u_idx(i, j): return i * m + j
    def v_idx(i, j): return i * p + j
    def w_idx(i, j): return j * n + i

    # 23 terms for 3x3x3
    # Indices: (u_i, u_j), (v_i, v_j), (w_i, w_j)
    # w_idx is j*n + i, so w_i is row, w_j is col?
    # C = A B. C[i,k] = sum_j A[i,j] B[j,k].
    # W index: k*n + i. So W[k, i] in matrix notation.
    # My w_idx(i, j) returns j*n + i. So w_idx(row, col) = col*n + row.
    # This matches W[k, i] if i is row of C and k is col of C.
    # W[k, i] is at index k*n + i.
    # So w_idx(i, k) should be k*n + i.
    # My function: w_idx(i, j) = j*n + i.
    # So w_idx(row_C, col_C) = col_C*n + row_C.
    # This is correct.

    terms_333 = [
        ((0,0), (0,0), (0,0)),
        ((0,1), (1,0), (0,0)),
        ((1,0), (0,1), (0,0)),
        ((1,1), (1,1), (0,0)),
        ((0,2), (2,0), (0,0)),
        ((2,0), (0,2), (0,0)),
        ((2,2), (2,2), (0,0)),
        ((0,0), (1,1), (1,0)),
        ((0,1), (0,1), (1,0)),
        ((1,0), (1,0), (1,0)),
        ((1,1), (0,0), (1,0)),
        ((0,2), (2,1), (1,0)),
        ((2,0), (1,2), (1,0)),
        ((2,2), (1,2), (1,0)),
        ((0,0), (2,2), (2,0)),
        ((0,2), (0,2), (2,0)),
        ((2,0), (2,0), (2,0)),
        ((2,2), (0,0), (2,0)),
        ((0,1), (2,1), (2,0)),
        ((1,0), (1,2), (2,0)),
        ((1,1), (2,2), (2,0)),
        ((1,2), (0,1), (0,1)),
        ((2,1), (1,0), (1,1))
    ]
    
    # 9 terms for 3x3x1
    # A (3x3) B (3x1) -> C (3x1)
    # C[i,0] = sum_j A[i,j] B[j,0]
    # Terms: (i,j), (j,0), (i,0)
    terms_331 = [
        ((0,0), (0,0), (0,0)),
        ((0,1), (1,0), (0,0)),
        ((0,2), (2,0), (0,0)),
        ((1,0), (0,0), (1,0)),
        ((1,1), (1,0), (1,0)),
        ((1,2), (2,0), (1,0)),
        ((2,0), (0,0), (2,0)),
        ((2,1), (1,0), (2,0)),
        ((2,2), (2,0), (2,0))
    ]

    # Add 3x3x3 terms for B[:, 0:3]
    for (ui, uj), (vi, vj), (wi, wj) in terms_333:
        U_vec = [0]*(n*m); V_vec = [0]*(m*p); W_vec = [0]*(p*n)
        U_vec[u_idx(ui, uj)] = 1
        V_vec[v_idx(vi, vj)] = 1
        W_vec[w_idx(wi, wj)] = 1
        add_rank1_term(U_vec, V_vec, W_vec)

    # Add 3x3x3 terms for B[:, 3:6]
    for (ui, uj), (vi, vj), (wi, wj) in terms_333:
        U_vec = [0]*(n*m); V_vec = [0]*(m*p); W_vec = [0]*(p*n)
        U_vec[u_idx(ui, uj)] = 1
        V_vec[v_idx(vi, vj + 3)] = 1
        W_vec[w_idx(wi, wj + 3)] = 1
        add_rank1_term(U_vec, V_vec, W_vec)

    # Add 3x3x1 terms for B[:, 6:7]
    for (ui, uj), (vi, vj), (wi, wj) in terms_331:
        U_vec = [0]*(n*m); V_vec = [0]*(m*p); W_vec = [0]*(p*n)
        U_vec[u_idx(ui, uj)] = 1
        V_vec[v_idx(vi, vj + 6)] = 1
        W_vec[w_idx(wi, wj + 6)] = 1
        add_rank1_term(U_vec, V_vec, W_vec)

    # Verification
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_true = A_test @ B_test
    C_comp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            val = 0.0
            for j in range(m):
                for r in range(rank):
                    val += U_arr[i*m + j, r] * A_test[i, j] * V_arr[j*p + k, r] * B_test[j, k] * W_arr[k*n + i, r]
            C_comp[i, k] = val
            
    if not np.allclose(C_comp, C_true, atol=1e-9):
        # Fallback to rank 60
        rank = 60
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        
        def add_strassen(a_base_i, b_base_i, c_base_i, k_base):
            nonlocal col
            if col >= rank: return
            a00 = u_idx(a_base_i, 0); a01 = u_idx(a_base_i, 1)
            a10 = u_idx(a_base_i + 1, 0); a11 = u_idx(a_base_i + 1, 1)
            b00 = v_idx(b_base_i, k_base); b01 = v_idx(b_base_i, k_base + 1)
            b10 = v_idx(b_base_i + 1, k_base); b11 = v_idx(b_base_i + 1, k_base + 1)
            c00 = w_idx(c_base_i, k_base); c01 = w_idx(c_base_i, k_base + 1)
            c10 = w_idx(c_base_i + 1, k_base); c11 = w_idx(c_base_i + 1, k_base + 1)
            
            terms = [
                (a00, b00, c00, 1, 1, 1), (a11, b00, c00, 1, 1, 1),
                (a10, b00, c10, 1, 1, 1), (a11, b00, c10, 1, 1, -1),
                (a00, b01, c01, 1, 1, 1), (a00, b11, c01, 1, -1, 1),
                (a11, b10, c00, 1, 1, 1), (a11, b00, c00, 1, -1, 1),
                (a00, b11, c00, 1, 1, -1), (a01, b11, c00, 1, 1, 1),
                (a10, b00, c11, 1, 1, 1), (a00, b00, c11, -1, 1, 1),
                (a01, b10, c00, 1, 1, 1), (a11, b10, c00, -1, 1, 1)
            ]
            # This fallback is complex, just use schoolbook
        # Simple schoolbook fallback
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
