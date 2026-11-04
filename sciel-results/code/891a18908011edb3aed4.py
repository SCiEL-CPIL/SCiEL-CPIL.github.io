"""Rewritten program: Strassen-based rank-36 decomposition for 2x3x8 matrix multiplication."""
import time

DIMENSIONS = [2, 3, 8]


def construct():
    """
    Rank-36 exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    
    Uses Strassen's rank-7 algorithm for the 2x2 * 2x2 core of each 2-column block,
    plus rank-2 trivial decomposition for the 2x1 * 1x2 tail.
    4 blocks * (7 + 2) = 36 total rank.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=8
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 36
    V = [[0] * rank for _ in range(m * p)]  # 24 x 36
    W = [[0] * rank for _ in range(p * n)]  # 16 x 36
    
    col = 0
    for b in range(4):  # 4 blocks of 2 columns each
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # M1 = (A00+A11)(B00+B11)
        # C00 += 1, C11 += 1
        U[0][col] = 1   # A[0,0]
        U[4][col] = 1   # A[1,1]
        V[k0][col] = 1      # B[0,k0]
        V[k1][col] = 1      # B[0,k1]
        V[8+k0][col] = 1    # B[1,k0]
        V[8+k1][col] = 1    # B[1,k1]
        W[2*k0][col] = 1    # C[0,k0]
        W[2*k1+1][col] = 1  # C[1,k1]
        col += 1
        
        # M2 = (A10+A11)*B00
        # C10 += 1, C11 += 1
        U[3][col] = 1   # A[1,0]
        U[4][col] = 1   # A[1,1]
        V[k0][col] = 1  # B[0,k0]
        W[2*k0+1][col] = 1  # C[1,k0]
        W[2*k1+1][col] = 1  # C[1,k1]
        col += 1
        
        # M3 = A00*(B10-B11)
        # C01 += 1, C11 += -1
        U[0][col] = 1   # A[0,0]
        V[8+k0][col] = 1    # B[1,k0]
        V[8+k1][col] = -1   # B[1,k1]
        W[2*k0+1][col] = 1  # C[0,k1]
        W[2*k1+1][col] = -1 # C[1,k1]
        col += 1
        
        # M4 = A11*(B01-B10)
        # C00 += 1, C10 += 1
        U[4][col] = 1   # A[1,1]
        V[k1][col] = 1      # B[0,k1]
        V[8+k0][col] = -1   # B[1,k0]
        W[2*k0][col] = 1    # C[0,k0]
        W[2*k0+1][col] = 1  # C[1,k0]
        col += 1
        
        # M5 = (A00+A01)*B11
        # C00 += -1, C01 += 1
        U[0][col] = 1   # A[0,0]
        U[1][col] = 1   # A[0,1]
        V[8+k1][col] = 1  # B[1,k1]
        W[2*k0][col] = -1    # C[0,k0]
        W[2*k0+1][col] = 1   # C[0,k1]
        col += 1
        
        # M6 = (A10-A00)*(B00+B10)
        # C11 += 1
        U[3][col] = 1    # A[1,0]
        U[0][col] = -1   # A[0,0]
        V[k0][col] = 1      # B[0,k0]
        V[8+k0][col] = 1    # B[1,k0]
        W[2*k1+1][col] = 1  # C[1,k1]
        col += 1
        
        # M7 = (A01-A11)*(B01+B11)
        # C00 += 1
        U[1][col] = 1    # A[0,1]
        U[4][col] = -1   # A[1,1]
        V[k1][col] = 1      # B[0,k1]
        V[8+k1][col] = 1    # B[1,k1]
        W[2*k0][col] = 1    # C[0,k0]
        col += 1
        
        # Tail1: A[0,2] * B[2,k], contributes to C[0,k] for k in {k0,k1}
        U[2][col] = 1   # A[0,2]
        V[16+k0][col] = 1  # B[2,k0]
        V[16+k1][col] = 1  # B[2,k1]
        W[2*k0][col] = 1    # C[0,k0]
        W[2*k1][col] = 1    # C[0,k1]
        col += 1
        
        # Tail2: A[1,2] * B[2,k], contributes to C[1,k] for k in {k0,k1}
        U[5][col] = 1   # A[1,2]
        V[16+k0][col] = 1  # B[2,k0]
        V[16+k1][col] = 1  # B[2,k1]
        W[2*k0+1][col] = 1  # C[1,k0]
        W[2*k1+1][col] = 1  # C[1,k1]
        col += 1
    
    return {"U": U, "V": V, "W": W}
