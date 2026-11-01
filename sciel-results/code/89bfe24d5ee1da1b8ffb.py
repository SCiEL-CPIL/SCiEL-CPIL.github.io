"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 18
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen for k in {0, 1}
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    # r=0: M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[14][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # r=1: M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # r=2: M3 = A00(B10+B11)
    U[0][2] = 1
    V[13][2] = 1; V[14][2] = 1
    W[2][2] = 1; W[3][2] = 1
    
    # r=3: M4 = A11(B01+B10)
    U[3][3] = 1
    V[1][3] = 1; V[13][3] = 1
    W[0][3] = 1
    
    # r=4: M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[14][4] = 1
    W[0][4] = -1
    
    # r=5: M6 = (A10-A00)(B01+B11)
    U[2][5] = 1; U[0][5] = -1
    V[1][5] = 1; V[14][5] = 1
    W[1][5] = 1; W[3][5] = 1
    
    # r=6: M7 = (A01-A11)(B00+B10)
    U[1][6] = 1; U[3][6] = -1
    V[0][6] = 1; V[13][6] = 1
    W[0][6] = -1; W[2][6] = 1
    
    # Rank 1 decompositions for k in {2, ..., 12}
    for k in range(2, 13):
        r = k + 5
        for i in range(2):
            for j in range(2):
                U[i*m + j][r] = 1
                V[j*p + k][r] = 1
                W[k*n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
