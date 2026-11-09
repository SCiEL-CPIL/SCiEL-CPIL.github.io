import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=3
    # A is n x m
    # B is m x p
    # C is n x p
    
    # Rank is n*m*p = 18
    rank = n * m * p
    
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Indexing:
    # U row index: i*m + j  (i in 0..n-1, j in 0..m-1)
    # V row index: j*p + k  (j in 0..m-1, k in 0..p-1)
    # W row index: k*n + i  (k in 0..p-1, i in 0..n-1)
    # Note: Task says W[k*n+i, r]. This matches k*n+i.
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
