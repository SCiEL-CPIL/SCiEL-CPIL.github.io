import time
import numpy as np

DIMENSIONS = [2, 2, 12]

def construct():
    """
    Rank-42 exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    Uses 6 blocks of 2 columns. Each block uses Strassen (7 terms) for the 2x2 sub-matrix.
    Total rank = 6 * 7 = 42.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    num_blocks = p // 2  # 6
    rank_per_block = 7
    rank = num_blocks * rank_per_block  # 42
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 24 x 42
    
    for t in range(num_blocks):
        base = t * rank_per_block
        col0 = 2 * t
        col1 = 2 * t + 1
        
        # Strassen 7 terms for 2x2 * 2x2 (columns col0, col1)
        # M1 = (A00 + A11) * (B0,col0 + B1,col1)
        s = 0
        c = base + s
        U[0][c] = 1   # A00
        U[3][c] = 1   # A11
        V[0*p + col0][c] = 1  # B[0,col0]
        V[1*p + col1][c] = 1  # B[1,col1]
        W[col0*n + 0][c] = 1  # C[col0, 0]
        W[col1*n + 1][c] = 1  # C[col1, 1]
        
        # M2 = (A10 + A11) * B0,col0
        s = 1
        c = base + s
        U[2][c] = 1   # A10
        U[3][c] = 1   # A11
        V[0*p + col0][c] = 1  # B[0,col0]
        W[col0*n + 1][c] = 1  # C[col0, 1]
        W[col1*n + 1][c] = -1 # C[col1, 1]
        
        # M3 = A00 * (B0,col1 - B1,col1)
        s = 2
        c = base + s
        U[0][c] = 1   # A00
        V[0*p + col1][c] = 1   # B[0,col1]
        V[1*p + col1][c] = -1  # B[1,col1]
        W[col1*n + 0][c] = 1   # C[col1, 0]
        W[col1*n + 1][c] = 1   # C[col1, 1]
        
        # M4 = A11 * (B1,col0 - B0,col0)
        s = 3
        c = base + s
        U[3][c] = 1   # A11
        V[1*p + col0][c] = 1   # B[1,col0]
        V[0*p + col0][c] = -1  # B[0,col0]
        W[col0*n + 0][c] = 1   # C[col0, 0]
        W[col0*n + 1][c] = 1   # C[col0, 1]
        
        # M5 = (A00 + A01) * B1,col1
        s = 4
        c = base + s
        U[0][c] = 1   # A00
        U[1][c] = 1   # A01
        V[1*p + col1][c] = 1  # B[1,col1]
        W[col0*n + 0][c] = -1 # C[col0, 0]
        W[col1*n + 0][c] = 1  # C[col1, 0]
        
        # M6 = (A10 - A00) * (B0,col0 + B0,col1)
        s = 5
        c = base + s
        U[2][c] = 1    # A10
        U[0][c] = -1   # A00
        V[0*p + col0][c] = 1  # B[0,col0]
        V[0*p + col1][c] = 1  # B[0,col1]
        W[col1*n + 1][c] = 1  # C[col1, 1]
        
        # M7 = (A01 - A11) * (B1,col0 + B1,col1)
        s = 6
        c = base + s
        U[1][c] = 1    # A01
        U[3][c] = -1   # A11
        V[1*p + col0][c] = 1  # B[1,col0]
        V[1*p + col1][c] = 1  # B[1,col1]
        W[col0*n + 0][c] = 1  # C[col0, 0]
    
    return {"U": U, "V": V, "W": W}
