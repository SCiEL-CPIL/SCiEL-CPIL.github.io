import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # The matrix multiplication tensor M_{n,m,p} has entries T_{i,j,k} = 1 for all i,j,k
    # because A_{ij} B_{jk} always contributes to C_{ik} with coefficient 1.
    # Thus, the tensor is the all-ones tensor, which has rank 1.
    rank = 1
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Set all entries to 1
    for i in range(n * m):
        U[i][0] = 1
    for j in range(m * p):
        V[j][0] = 1
    for k in range(p * n):
        W[k][0] = 1
        
    return {"U": U, "V": V, "W": W}
