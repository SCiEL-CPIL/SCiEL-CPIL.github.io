import numpy as np

def construct():
    n, m, p = 2, 3, 9
    rank = n * m * p  # 54
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 54
    V = [[0] * rank for _ in range(m * p)]  # 27 x 54
    W = [[0] * rank for _ in range(p * n)]  # 18 x 54
    
    col = 0
    for k in range(p):
        for i in range(n):
            for j in range(m):
                # U index: row i*m + j (input matrix A, n x m)
                U[i * m + j][col] = 1
                # V index: row j*p + k (input matrix B, m x p)
                V[j * p + k][col] = 1
                # W index: row k*n + i (output matrix C, p x n, trace convention)
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
