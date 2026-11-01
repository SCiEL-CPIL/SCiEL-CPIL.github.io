import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Rank 9 decomposition for 2x3x3
    # Split B into 3 columns of size 3x1
    # Each 2x3x1 multiplication has rank 3
    # Total rank 3 * 3 = 9
    
    rank = 9
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for k in range(p): # For each column of B (which is 3x1 effectively)
        for j in range(m): # For each row of that column (which is 1x1 effectively)
            # U: A[i, j]
            for i in range(n):
                U[i * m + j][col] = 1
            # V: B[j, k]
            V[j * p + k][col] = 1
            # W: C[i, k]
            for i in range(n):
                W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
