import time
import sys

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=5, p=7
    
    # Schoolbook decomposition: rank = n*m*p = 105
    # For each (i, j, k) triple, add a rank-1 term:
    #   U[i*m + j, t] = 1
    #   V[j*p + k, t] = 1
    #   W[k*n + i, t] = 1
    # This gives C[i,k] = sum_j A[i,j] * B[j,k] correctly.
    
    rank = n * m * p  # 105
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][t] = 1
                V[j * p + k][t] = 1
                W[k * n + i][t] = 1
                t += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
