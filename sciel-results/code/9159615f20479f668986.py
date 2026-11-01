import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Naive matrix multiplication decomposition
    # Each term t corresponds to a unique (i, j, k) index triple
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # Term t: A[i, j] * B[j, k] contributes to C[i, k]
    
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: row i, col j of A. Flatten index: i * m + j
                u_idx = i * m + j
                U[u_idx][t] = 1
                
                # V index: row j, col k of B. Flatten index: j * p + k
                v_idx = j * p + k
                V[v_idx][t] = 1
                
                # W index: col k, row i of C. 
                # Convention W[k*n + i, r]
                w_idx = k * n + i
                W[w_idx][t] = 1
                
                t += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
