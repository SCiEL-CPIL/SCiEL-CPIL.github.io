import numpy as np
import time

def construct():
    """
    Constructs the schoolbook tensor decomposition for 4x4 * 4x9 matrix multiplication.
    Rank is 4 * 4 * 9 = 144.
    
    The output format is:
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    
    Indexing convention:
    U[i*m + j, r]
    V[j*p + k, r]
    W[k*n + i, r]
    """
    n, m, p = 4, 4, 9
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                
                U[a, col] = 1.0
                V[b, col] = 1.0
                W[c, col] = 1.0
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
