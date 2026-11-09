"""Matrix multiplication 3x3 x 3x8 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x3 x 3x8 matrix multiplication.
    
    The 3x3 x 3x8 multiplication decomposes into 8 independent 3x3 x 3x1 
    matrix-vector products. The rank of M_{3,3,1} is 6, so total rank = 8*6 = 48.
    
    Rank-6 decomposition for 3x3 x 3x1 (C = A*b):
    c1 = a11*b1 + a12*b2 + a13*b3
    c2 = a21*b1 + a22*b2 + a23*b3
    c3 = a31*b1 + a32*b2 + a33*b3
    
    Using these 6 bilinear terms:
    T1 = a11*b1
    T2 = a22*b2
    T3 = a33*b3
    T4 = (a12 + a21)*b2  -- no, let me be more careful
    
    Actually, a valid rank-6 decomposition:
    T1 = (a11 + a22 + a33) * b1
    T2 = (a12 + a23 + a31) * b2
    T3 = (a13 + a21 + a32) * b3
    T4 = (a11 + a12 + a13) * (b1 + b2 + b3)
    T5 = (a21 + a22 + a23) * (b1 + b2 + b3)
    T6 = (a31 + a32 + a33) * (b1 + b2 + b3)
    -- This doesn't work either.
    
    Let me use the simple schoolbook for each column: rank = 3*3*8 = 72.
    This is guaranteed correct.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
