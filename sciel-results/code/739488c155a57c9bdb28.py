import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for matrix multiplication
    of dimensions n x m x p.
    Currently implements the schoolbook algorithm with rank n*m*p.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros using NumPy
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
