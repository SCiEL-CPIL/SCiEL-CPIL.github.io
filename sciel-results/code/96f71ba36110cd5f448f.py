"""Rewritten program for 2x2 matrix multiplication rank minimization."""
import numpy as np
from itertools import product, permutations

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the optimal rank-7 Strassen decomposition for 2x2 matrix multiplication.
    
    The tensor rank of 2x2 matrix multiplication is 7.
    U shape: (n*m, rank) -> (4, 7)
    V shape: (m*p, rank) -> (4, 7)
    W shape: (p*n, rank) -> (4, 7)
    
    Indexing:
    U[i*m + a, r] corresponds to coefficient of A[i, a] in the r-th product.
    V[a*p + k, r] corresponds to coefficient of B[a, k] in the r-th product.
    W[k*n + i, r] corresponds to coefficient of the r-th product in C[i, k].
    """
    n, m, p = 2, 2, 2
    rank = 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products:
    # M0: (A00 + A11)(B00 + B11)
    U[0*2+0][0] = 1; U[1*2+1][0] = 1
    V[0*2+0][0] = 1; V[1*2+1][0] = 1
    
    # M1: (A10 + A11)B00
    U[1*2+0][1] = 1; U[1*2+1][1] = 1
    V[0*2+0][1] = 1
    
    # M2: A00(B01 - B11)
    U[0*2+0][2] = 1
    V[0*2+1][2] = 1; V[1*2+1][2] = -1
    
    # M3: A11(B10 - B00)
    U[1*2+1][3] = 1
    V[1*2+0][3] = 1; V[0*2+0][3] = -1
    
    # M4: (A00 + A01)B11
    U[0*2+0][4] = 1; U[0*2+1][4] = 1
    V[1*2+1][4] = 1
    
    # M5: (A10 - A00)(B00 + B01)
    U[1*2+0][5] = 1; U[0*2+0][5] = -1
    V[0*2+0][5] = 1; V[0*2+1][5] = 1
    
    # M6: (A01 - A11)(B10 + B11)
    U[0*2+1][6] = 1; U[1*2+1][6] = -1
    V[1*2+0][6] = 1; V[1*2+1][6] = 1
    
    # Output reconstruction:
    # C[0,0] = M0 + M3 - M4 + M6
    W[0*2+0][0] = 1; W[0*2+0][3] = 1; W[0*2+0][4] = -1; W[0*2+0][6] = 1
    
    # C[0,1] = M2 + M4
    W[1*2+0][2] = 1; W[1*2+0][4] = 1
    
    # C[1,0] = M1 + M3
    W[0*2+1][1] = 1; W[0*2+1][3] = 1
    
    # C[1,1] = M0 - M1 + M2 + M5
    W[1*2+1][0] = 1; W[1*2+1][1] = -1; W[1*2+1][2] = 1; W[1*2+1][5] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
