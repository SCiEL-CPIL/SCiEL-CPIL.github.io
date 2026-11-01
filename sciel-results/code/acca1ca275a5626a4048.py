import time
import numpy as np

def construct():
    """
    Construct a rank-7 tensor decomposition for 2x2 matrix multiplication
    using Strassen's algorithm.
    """
    n, m, p = 2, 2, 2
    rank = 7
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # (4, 7)
    V = [[0] * rank for _ in range(m * p)]  # (4, 7)
    W = [[0] * rank for _ in range(p * n)]  # (4, 7)
    
    # Strassen's 7 products:
    # M1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    
    # M2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    
    # M3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    
    # M4 = d(g-b)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    
    # M5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    
    # M6 = (c-d)(g+h)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    
    # M7 = (a-c)(b+d)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    
    # W factors (reconstructed to match the above M's)
    # W[k*n+i, r]
    # P11 = M1 + M4 - M5 + M7
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    # P12 = M2 + M4
    W[1][1] = 1; W[1][3] = 1
    # P21 = M3 + M5
    W[2][2] = 1; W[2][4] = 1
    # P22 = M1 - M2 + M6 + M7
    W[3][0] = 1; W[3][1] = -1; W[3][5] = 1; W[3][6] = 1
    
    return {"U": U, "V": V, "W": W}
