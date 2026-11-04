import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS

    # Rank is n*m*p for schoolbook
    rank = n * m * p

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Schoolbook decomposition
    # T[i,j,k] = 1 for all i,j,k
    # U[i*m+j, r] = 1 if r == i*m*p + j*p + k? No.
    # We need sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1.
    # Let r index the triple (i,j,k).
    # U[i*m+j, r] = 1 if r corresponds to (i,j,k) for any k? No.
    # If r = i*m*p + j*p + k, then:
    # U[i*m+j, r] = 1.
    # V[j*p+k, r] = 1.
    # W[k*n+i, r] = 1.
    # Product is 1.
    # For other r', if i',j',k' != i,j,k, at least one factor is 0.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1

    return {"U": U, "V": V, "W": W}
