"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Construct a tensor decomposition for 3x3 * 3x6 matrix multiplication.
    
    The multiplication tensor T has T[a,b,c] = 1 iff there exist i,j,k such that:
      a = i*m + j  (i in [0,n), j in [0,m))
      b = j*p + k  (j in [0,m), k in [0,p))
      c = k*n + i  (k in [0,p), i in [0,n))
    
    For n=3, m=3, p=6:
      a = 3*i + j
      b = 3*j + k  where j in {0,1,2}, k in {0,...,5}
      c = 6*k + i  where k in {0,...,5}, i in {0,1,2}
    
    Strategy: Use schoolbook decomposition with rank n*m*p = 3*3*6 = 54.
    Each term corresponds to a unique (i,j,k) triple.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    
    rank = n * m * p  # 54 for schoolbook
    
    U = [[0.0] * rank for _ in range(n * m)]  # shape (9, 54)
    V = [[0.0] * rank for _ in range(m * p)]  # shape (18, 54)
    W = [[0.0] * rank for _ in range(p * n)]  # shape (18, 54)
    
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][t] = 1.0
                V[j * p + k][t] = 1.0
                W[k * n + i][t] = 1.0
                t += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
