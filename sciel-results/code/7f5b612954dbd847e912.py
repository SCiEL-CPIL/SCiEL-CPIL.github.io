"""Matrix multiplication 4x5 by 5x5: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]
DEADLINE_S = 110.0


def construct():
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    t0 = time.time()
    
    # Schoolbook decomposition: rank = n*m*p = 100
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
    
    # Verify correctness: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] 
    # should equal delta_{j,j'} for the multiplication tensor
    # Actually, the condition is: for all A, B, C = A@B
    # C[k, i] = sum_j A[i, j] * B[j, k]
    # W index is k*n + i, so W[k*n+i, r] corresponds to output C[k, i]
    
    # The tensor condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    # This is exactly the schoolbook: each term picks out one (i,j,k) triple
    
    return {"U": U, "V": V, "W": W}
