"""Matrix multiplication 2,3,9 - rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The rank is n*m*p = 54.
    # We construct the decomposition by treating the problem as n independent
    # 1 x m x p multiplications.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    # Iterate over each row of the first matrix (i)
    for i in range(n):
        # Iterate over each column of the result (k)
        for k in range(p):
            # Iterate over the intermediate dimension (j)
            for j in range(m):
                # U index: i*m + j
                U[i * m + j, col] = 1
                # V index: j*p + k
                V[j * p + k, col] = 1
                # W index: k*n + i
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
