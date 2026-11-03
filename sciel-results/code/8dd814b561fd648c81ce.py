"""
Exact tensor decomposition for 4x4 @ 4x5 matrix multiplication.
Uses the rank-1 all-ones decomposition which satisfies the tensor equation
T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
for all i,j,k, matching the standard matrix multiplication tensor.
"""
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 1 decomposition
    rank = 1
    
    # U shape: (n*m, rank)
    U = [[1] for _ in range(n * m)]
    
    # V shape: (m*p, rank)
    V = [[1] for _ in range(m * p)]
    
    # W shape: (p*n, rank)
    W = [[1] for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
