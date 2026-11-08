import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a valid tensor decomposition for 2x3x6 matrix multiplication.
    Uses the schoolbook decomposition with rank n*m*p = 36.
    Returns U, V, W as NumPy arrays to ensure compatibility with evaluators 
    that expect array types.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize arrays with zeros
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U is indexed by (i, j) flattened to i*m + j
                U[i * m + j, column] = 1.0
                # V is indexed by (j, k) flattened to j*p + k
                V[j * p + k, column] = 1.0
                # W is indexed by (k, i) flattened to k*n + i (trace convention)
                W[k * n + i, column] = 1.0
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
