"""Matrix multiplication 2x2 by 2x7 - rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 7]

def construct():
    """
    Construct a tensor decomposition for M(2,2,7).
    
    A is 2x2, B is 2x7, C = A*B is 2x7.
    W[k*n+i, r] gives the coefficient for C[i,k] (n=2, so W[2*k+i, r]).
    
    Schoolbook: 28 rank-1 terms, one per (i,j,t) triple.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=7
    deadline = time.time() + 140  # safety margin before 150s kill

    rank = n * m * p  # 28
    U = [[0.0] * rank for _ in range(n * m)]   # (4, 28)
    V = [[0.0] * rank for _ in range(m * p)]   # (14, 28)
    W = [[0.0] * rank for _ in range(p * n)]   # (14, 28)

    r = 0
    for i in range(n):
        for j in range(m):
            for t in range(p):
                U[i * m + j][r] = 1.0
                V[j * p + t][r] = 1.0
                W[t * n + i][r] = 1.0
                r += 1

    return {"U": U, "V": V, "W": W}
