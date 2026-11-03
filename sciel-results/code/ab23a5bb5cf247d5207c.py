"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct an exact tensor decomposition for 2x3 * 3x9 matrix multiplication.
    
    We use the schoolbook decomposition of rank n*m*p = 54.
    For each (i, j, k) in [0,n) x [0,m) x [0,p):
      U[i*m + j, r] = 1
      V[j*p + k, r] = 1
      W[k*n + i, r] = 1
    where r = i*m*p + j*p + k.
    
    Verification:
    sum_r U[a,r] V[b,r] W[c,r]
    = sum_{i,j,k} [a == i*m+j] [b == j*p+k] [c == k*n+i]
    = 1 iff there exist i,j,k with a=i*m+j, b=j*p+k, c=k*n+i
    which is exactly the matrix multiplication tensor condition.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
