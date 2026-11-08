import numpy as np

def construct():
    n, m, p = 4, 4, 6
    
    # Standard Kronecker product decomposition
    # Rank = n * m * p
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    k = 0
    for i in range(n):
        for j in range(m):
            for l in range(p):
                # U: A[i, j]
                U[i * m + j][k] = 1
                # V: B[j, l]
                V[j * p + l][k] = 1
                # W: C[i, l] (Trace convention: W[l*n + i])
                W[l * n + i][k] = 1
                k += 1
                
    return {"U": U, "V": V, "W": W}
