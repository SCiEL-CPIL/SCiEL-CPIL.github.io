import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 110.0

    # Strassen rank-7 decomposition for 2x2 matrix multiplication.
    # U2 row index = i*2+j (A entry), V2 row index = j*2+k (B entry), W2 row index = k*2+i (C entry).
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)

    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)

    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)

    # Build 4x4 factors from 2x2 Strassen factors using Kronecker product logic
    # U4[i*4+j, c1*7+c2] = U2[i_hi*2+j_hi, c1] * U2[i_lo*2+j_lo, c2]
    # where i = 2*i_hi + i_lo, j = 2*j_hi + j_lo
    
    # Use Kronecker product directly: reshape and multiply
    # U2 is 4x7, we want to build 16x49
    # For each pair of rows (r1, r2) in U2, the corresponding row in U4 is U2[r1,:] ⊗ U2[r2,:]
    
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    # Precompute row pairs
    row_pairs_u = []
    for i in range(4):
        for j in range(4):
            i_hi, i_lo = i // 2, i % 2
            j_hi, j_lo = j // 2, j % 2
            row_pairs_u.append((i*4+j, i_hi*2+j_hi, i_lo*2+j_lo))
    
    for idx_u, r_hi, r_lo in row_pairs_u:
        U4[idx_u, :] = np.kron(U2[r_hi, :], U2[r_lo, :])
    
    row_pairs_v = []
    for j in range(4):
        for k in range(4):
            j_hi, j_lo = j // 2, j % 2
            k_hi, k_lo = k // 2, k % 2
            row_pairs_v.append((j*4+k, j_hi*2+k_hi, j_lo*2+k_lo))
    
    for idx_v, r_hi, r_lo in row_pairs_v:
        V4[idx_v, :] = np.kron(V2[r_hi, :], V2[r_lo, :])
    
    row_pairs_w = []
    for k in range(4):
        for i in range(4):
            k_hi, k_lo = k // 2, k % 2
            i_hi, i_lo = i // 2, i % 2
            row_pairs_w.append((k*4+i, k_hi*2+i_hi, k_lo*2+i_lo))
    
    for idx_w, r_hi, r_lo in row_pairs_w:
        W4[idx_w, :] = np.kron(W2[r_hi, :], W2[r_lo, :])

    if time.time() - start_time > deadline:
        # Fallback: schoolbook 4x4 (rank 64)
        n, m, p = 4, 4, 4
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }
