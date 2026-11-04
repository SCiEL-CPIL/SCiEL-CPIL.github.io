import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    # Verification
    # The schoolbook decomposition is exact.
    # T[i,j,k] = 1 if indices match, 0 otherwise.
    # Sum over c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # Only the term where c corresponds to (i,j,k) is 1*1*1=1. All others are 0.
    # So the sum is 1.
    
    return {"U": U, "V": V, "W": W}
