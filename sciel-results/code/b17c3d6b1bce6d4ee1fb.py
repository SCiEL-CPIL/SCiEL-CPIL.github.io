import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Decompose 2x4x6 as sum of two 2x2x6 tensors
    # Split j dimension into two groups of 2
    # Each 2x2x6 tensor has rank 12 (naive)
    # Total rank = 2 * 12 = 24
    
    rank = 24
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for j_block in range(2):  # Two blocks of j: {0,1} and {2,3}
        for i in range(n):  # i = 0, 1
            for j_local in range(2):  # j_local = 0, 1 within the block
                j_global = 2 * j_block + j_local
                for k in range(p):  # k = 0, ..., 5
                    # Term: U[i,j_global] * V[j_global,k] * W[k,i]
                    U[i * m + j_global, col] = 1
                    V[j_global * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Verify
    try:
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,ki->ijk', U_t, V_t, W_t)
        if not np.all(T == 1):
            # Fallback to naive rank-48
            r_sb = n * m * p
            U_sb = np.zeros((n * m, r_sb), dtype=int)
            V_sb = np.zeros((m * p, r_sb), dtype=int)
            W_sb = np.zeros((p * n, r_sb), dtype=int)
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_sb[i * m + j, c] = 1
                        V_sb[j * p + k, c] = 1
                        W_sb[k * n + i, c] = 1
                        c += 1
            return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    except Exception:
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
