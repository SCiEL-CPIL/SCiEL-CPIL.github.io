import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank is sum of ranks of sub-problems
    # 1. i in 0..3, j in 0..3, k in 0..3 (4x4x4) -> rank 16
    # 2. i in 0..3, j in 0..3, k in 4..4 (4x4x1) -> rank 16
    # 3. i in 0..3, j in 4..4, k in 0..3 (4x1x4) -> rank 16
    # 4. i in 0..3, j in 4..4, k in 4..4 (4x1x1) -> rank 4
    # Total rank = 52
    
    rank = 52
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Block 1: j in 0..3, k in 0..3
    for i in range(n):
        for j in range(4):
            for k in range(4):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
                
    # Block 2: j in 0..3, k in 4
    for i in range(n):
        for j in range(4):
            for k in [4]:
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1

    # Block 3: j in 4, k in 0..3
    for i in range(n):
        for j in [4]:
            for k in range(4):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1

    # Block 4: j in 4, k in 4
    for i in range(n):
        for j in [4]:
            for k in [4]:
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
