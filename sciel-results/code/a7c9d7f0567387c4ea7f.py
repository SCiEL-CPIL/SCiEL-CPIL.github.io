import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 25  # One term per (i,k) pair, summing over j internally
    
    # U shape: (n*m, rank) = (25, 25)
    # V shape: (m*p, rank) = (25, 25)
    # W shape: (p*n, rank) = (25, 25)
    
    # Indexing conventions:
    # A[i,j] stored at U index i*m + j
    # B[j,k] stored at V index j*p + k
    # C[i,k] stored at W index k*n + i  (trace convention W[k*n+i, r])
    
    # For each (i, k) pair, we create one rank-1 term:
    # The product C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Term r = i*p + k:
    #   U[i*m + j, r] = 1 for all j  (extract row i of A)
    #   V[j*p + k, r] = 1 for all j  (extract column k of B)
    #   W[k*n + i, r] = 1            (place into C[i,k])
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for k in range(p):
            # U: for each j, U[i*m+j][r] = 1
            for j in range(m):
                U[i * m + j][r] = 1.0
            
            # V: for each j, V[j*p+k][r] = 1
            for j in range(m):
                V[j * p + k][r] = 1.0
            
            # W: W[k*n+i][r] = 1
            W[k * n + i][r] = 1.0
            
            r += 1
    
    return {"U": U, "V": V, "W": W}
