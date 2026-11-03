"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = 4, 4, 7
    rank = n * m * p  # 112 for schoolbook
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Schoolbook decomposition
    # T[i,j,k] = 1 for all i,j,k
    # Decomposition: sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{k,i}
    # This has rank n*m*p.
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][idx] = 1
                V[j*p + k][idx] = 1
                W[k*n + i][idx] = 1
                idx += 1
    
    return {'U': U, 'V': V, 'W': W}
# EVOLVE-BLOCK-END
