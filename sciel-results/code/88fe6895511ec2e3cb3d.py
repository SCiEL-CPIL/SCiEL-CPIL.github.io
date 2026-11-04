import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    """
    Returns the schoolbook tensor decomposition for matrix multiplication of 
    (2x4) * (4x7) -> (2x7).
    Rank is n*m*p = 56.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][column] = 1
                # V index: j*p + k
                V[j * p + k][column] = 1
                # W index: k*n + i (Official trace convention W[k*n+i, r])
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
