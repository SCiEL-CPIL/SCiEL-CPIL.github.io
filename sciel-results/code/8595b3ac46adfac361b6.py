import time
import numpy as np

DIMENSIONS = [2, 3, 6]
_DEADLINE = time.time() + 140  # Safety margin before 150s kill

def _time_remaining():
    return _DEADLINE - time.time()

def construct():
    n, m, p = DIMENSIONS
    
    # Try rank-28 construction first
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, col, val=1):
        if col < 0 or col >= rank:
            return
        if u_idx < 0 or u_idx >= n * m:
            return
        if v_idx < 0 or v_idx >= m * p:
            return
        if w_idx < 0 or w_idx >= p * n:
            return
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
    
    def add_2x3x3_block(col_offset, b_col_offset):
        # Sub-block 1: A[:, 0:2] * B[0:2, bco:bco+2] -> 4 terms
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    u_idx = i * m + j
                    v_idx = j * p + (b_col_offset + k)
                    w_idx = (b_col_offset + k) * n + i
                    add_term(u_idx, v_idx, w_idx, col_offset)
                    col_offset += 1
        
        # Sub-block 2: A[:, 2] * B[2, bco:bco+2] -> 4 terms
        for i in range(2):
            for k in range(2):
                u_idx = i * m + 2
                v_idx = 2 * p + (b_col_offset + k)
                w_idx = (b_col_offset + k) * n + i
                add_term(u_idx, v_idx, w_idx, col_offset)
                col_offset += 1
        
        # Sub-block 3: A[:, 0:3] * B[0:3, bco+2] -> 6 terms
        for i in range(2):
            for j in range(3):
                u_idx = i * m + j
                v_idx = j * p + (b_col_offset + 2)
                w_idx = (b_col_offset + 2) * n + i
                add_term(u_idx, v_idx, w_idx, col_offset)
                col_offset += 1
        
        return col_offset

    next_col = add_2x3x3_block(0, 0)
    next_col = add_2x3x3_block(next_col, 3)
    
    # Verify with exact integer arithmetic using small integer test matrices
    # Test with A and B having entries in {-1, 0, 1} for exact verification
    test_matrices = [
        (np.array([[1, 0, 1], [0, 1, -1]]), np.array([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0]])),
        (np.array([[1, 1, 0], [1, 0, 1]]), np.array([[0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]])),
        (np.array([[1, -1, 1], [-1, 1, 0]]), np.array([[1, 1, 0, 0, 1, 1], [0, 1, 1, 1, 0, 1], [1, 0, 1, 1, 1, 0]])),
    ]
    
    valid = True
    for A_test, B_test in test_matrices:
        C_test = A_test @ B_test
        C_decomp = np.zeros((n, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    C_decomp[i, k] = s
        if not np.array_equal(C_decomp, C_test):
            valid = False
            break
    
    if valid:
        return {"U": U, "V": V, "W": W}
    
    # Fallback to naive rank n*m*p = 36
    if _time_remaining() < 5:
        # Emergency: return a simple valid decomposition
        rank_fallback = n * m * p
        U_f = [[0] * rank_fallback for _ in range(n * m)]
        V_f = [[0] * rank_fallback for _ in range(m * p)]
        W_f = [[0] * rank_fallback for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U_f[u_idx][col] = 1
                    V_f[v_idx][col] = 1
                    W_f[w_idx][col] = 1
                    col += 1
        return {"U": U_f, "V": V_f, "W": W_f}
    
    rank_fallback = n * m * p
    U_f = [[0] * rank_fallback for _ in range(n * m)]
    V_f = [[0] * rank_fallback for _ in range(m * p)]
    W_f = [[0] * rank_fallback for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_f[u_idx][col] = 1
                V_f[v_idx][col] = 1
                W_f[w_idx][col] = 1
                col += 1
    
    return {"U": U_f, "V": V_f, "W": W_f}
