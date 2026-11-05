import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank is m (the shared dimension)
    rank = m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each shared dimension index k
    for k in range(m):
        # U indices: for each row i of A, the element A[i][k] is at flat index i*m + k
        for i in range(n):
            U[i * m + k][k] = 1
            
        # V indices: for each column j of B, the element B[k][j] is at flat index k*p + j
        for j in range(p):
            V[k * p + j][k] = 1
            
        # W indices: for each output element C[i][j], the index is j*n + i (trace convention W[k*n+i, r])
        # The term A[i][k] * B[k][j] contributes to C[i][j]
        for i in range(n):
            for j in range(p):
                W[j * n + i][k] = 1
                
    return {"U": U, "V": V, "W": W}
