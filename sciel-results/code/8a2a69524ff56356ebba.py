import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as NumPy arrays
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Construct the schoolbook decomposition using vectorized operations
    # for each term (i, j, k), we have a rank-1 component
    # U[i*m + j, c] = 1
    # V[j*p + k, c] = 1
    # W[k*n + i, c] = 1
    
    # We can iterate over c and set the non-zero elements
    # Or better, iterate over i, j, k and map to c
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = (i * m + j) * p + k
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
