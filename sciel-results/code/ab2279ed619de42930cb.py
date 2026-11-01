import time
import numpy as np

def _schoolbook(n=2, m=2, p=12):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _check(U, V, W, n, m, p, tol=1e-8):
    R = len(U[0]) if len(U) > 0 else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - (1.0 if j == i else 0.0)) > tol:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_t = time.time()
    deadline = start_t + 100

    n, m, p = 2, 2, 12
    fallback = _schoolbook(n, m, p)

    # Strassen rank-7 decomposition for 2x2x2 matrix multiplication
    U_strassen = np.array([
        [ 1,  1,  0,  0,  1,  0, -1],
        [ 0,  1,  1,  0,  0,  1,  0],
        [ 0,  0,  1,  1,  0,  0,  1],
        [ 0,  0,  1,  1,  1, -1,  0]
    ], dtype=float)

    V_strassen = np.array([
        [ 1,  0,  0,  1,  1,  1,  1],
        [ 0,  0,  1,  0,  0,  0,  1],
        [ 0,  1,  0,  0,  0,  1,  0],
        [ 1,  0,  0, -1,  1,  1,  0]
    ], dtype=float)

    W_strassen = np.array([
        [ 1,  0,  0,  0, -1, -1,  1],
        [ 0,  1,  1,  0,  0,  1,  0],
        [ 0,  0,  0,  1,  1,  0,  0],
        [ 0,  0,  0,  0,  0,  1,  1]
    ], dtype=float)

    def _is_valid_block(U_b, V_b, W_b, ns, ms, ps):
        for i in range(ns):
            for j in range(ms):
                for k in range(ps):
                    u_val = U_b[i*ms + j]
                    v_val = V_b[j*ps + k]
                    w_val = W_b[k*ns + i]
                    s = np.dot(u_val, v_val * w_val)
                    target = 1.0 if j == i else 0.0
                    if abs(s - target) > 1e-10:
                        return False
        return True

    if not _is_valid_block(U_strassen, V_strassen, W_strassen, 2, 2, 2):
        return fallback

    # Construct the full rank-42 decomposition
    num_blocks = p // 2
    block_rank = 7
    total_rank = num_blocks * block_rank

    U_full = np.zeros((n * m, total_rank), dtype=float)
    V_full = np.zeros((m * p, total_rank), dtype=float)
    W_full = np.zeros((p * n, total_rank), dtype=float)

    for block_idx in range(num_blocks):
        col_offset = block_idx * 2
        
        for r in range(block_rank):
            col = block_idx * block_rank + r
            
            U_full[:, col] = U_strassen[:, r]
            
            V_full[0*p + col_offset, col]     = V_strassen[0, r]
            V_full[0*p + col_offset + 1, col] = V_strassen[1, r]
            V_full[1*p + col_offset, col]     = V_strassen[2, r]
            V_full[1*p + col_offset + 1, col] = V_strassen[3, r]
            
            W_full[col_offset*n + 0, col]     = W_strassen[0, r]
            W_full[(col_offset+1)*n + 0, col] = W_strassen[1, r]
            W_full[col_offset*n + 1, col]     = W_strassen[2, r]
            W_full[(col_offset+1)*n + 1, col] = W_strassen[3, r]

    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    if _check(U_list, V_list, W_list, n, m, p):
        if time.time() < deadline:
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return fallback
# EVOLVE-BLOCK-END
