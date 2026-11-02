import numpy as np

def construct():
    n, m, p = 2, 2, 16
    rank = n * m * p
    
    # Generate indices for all (i, j, k) triples
    # c is the column index (rank dimension)
    c = np.arange(rank)
    
    # Decompose c into i, j, k
    # c = i * (m * p) + j * p + k
    k = c % p
    tmp = c // p
    j = tmp % m
    i = tmp // m
    
    # Calculate row indices
    row_u = i * m + j
    row_v = j * p + k
    row_w = k * n + i
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Scatter ones
    U[row_u, c] = 1
    V[row_v, c] = 1
    W[row_w, c] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
