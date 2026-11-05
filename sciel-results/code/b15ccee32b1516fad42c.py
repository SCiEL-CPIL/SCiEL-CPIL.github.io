"""Matrix multiplication 3x5x7 - attempting rank-minimal decomposition."""
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition: rank = 3*5*7 = 105
    # Using NumPy for efficient initialization and filling
    n, m, p = 3, 5, 7
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Iterate over all combinations of indices to set the 1s
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Calculate the column index for this term
                col = (i * m + j) * p + k
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
