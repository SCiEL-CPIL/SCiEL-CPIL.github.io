import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=5
    # Rank R = m * p = 25
    R = m * p
    
    # U shape (n*m, R) -> (10, 25)
    # V shape (m*p, R) -> (25, 25)
    # W shape (p*n, R) -> (10, 25)
    
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # r corresponds to (j, k)
    for j in range(m):
        for k in range(p):
            r = j * p + k
            
            # V[j*p + k][r] = 1
            V[j * p + k][r] = 1
            
            # U[i*m + j][r] = 1 for all i
            for i in range(n):
                U[i * m + j][r] = 1
                
            # W[k*n + i][r] = 1 for all i
            for i in range(n):
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
