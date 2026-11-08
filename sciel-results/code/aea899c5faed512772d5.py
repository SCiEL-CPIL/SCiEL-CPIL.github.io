import numpy as np

def construct():
    n, m, p = 4, 5, 5
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Standard matrix multiplication decomposition
    # T = sum_{i,j,k} A_{ij} B_{jk} C_{ik}
    # Rank = n*m*p
    # Term index t = i*m*p + j*p + k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                t = i * m * p + j * p + k
                # U row corresponds to A_{ij}
                U[i * m + j][t] = 1
                # V row corresponds to B_{jk}
                V[j * p + k][t] = 1
                # W row corresponds to C_{ik}
                # Official trace convention: W[k*n+i, r]
                W[k * n + i][t] = 1
                
    return {"U": U, "V": V, "W": W}
