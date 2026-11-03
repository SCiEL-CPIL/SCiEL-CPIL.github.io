import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition for 6x6 matrix multiplication
    # Rank = n * m * p = 216
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    idx = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                # U corresponds to A[i, k]
                U[i * m + k, idx] = 1
                # V corresponds to B[k, j]
                V[k * p + j, idx] = 1
                # W corresponds to C[i, j] but indexed as W[j * n + i]
                # based on the trace convention W[k*n+i] where k is the 'col' index of C (j)
                # and i is the 'row' index of C (i)
                W[j * n + i, idx] = 1
                idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
