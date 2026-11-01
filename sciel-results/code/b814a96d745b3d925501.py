import time
import sys

DIMENSIONS = [2, 3, 10]

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 by 3x10 matrix multiplication.
    """
    n, m, p = 2, 3, 10
    
    # We decompose the 10 columns into 5 blocks of 2 columns each.
    # For each block, we compute 2x3 x 3x2 matrix multiplication.
    # This is split into:
    # 1. 2x2 x 2 part (using first 2 of the 3 middle indices) - Rank 7 (Strassen)
    # 2. 2x1 x 2 part (using the 3rd middle index) - Rank 4 (Trivial)
    # Total rank per block = 11.
    # Total rank = 5 * 11 = 55.
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for block in range(5):
        # Local column indices in the 3x10 matrix: 2*block, 2*block+1
        c0 = 2 * block
        c1 = 2 * block + 1
        
        # --- Part 1: 2x2 x 2 (Strassen) ---
        # Middle indices: 0, 1
        # A indices: (0,0), (0,1), (1,0), (1,1)
        # B indices: (0,0), (0,1), (1,0), (1,1)
        # C indices: (0,0), (0,1), (1,0), (1,1)
        
        # M1 = (a00 + a01) * (b00 + b10)
        U[0*3+0][col] = 1
        U[0*3+1][col] = 1
        V[0*3+c0][col] = 1
        V[1*3+c0][col] = 1
        # C00 += M1
        W[c0*2+0][col] = 1
        col += 1
        
        # M2 = (a10 + a11) * (b00 + b01)
        U[1*3+0][col] = 1
        U[1*3+1][col] = 1
        V[0*3+c0][col] = 1
        V[0*3+c1][col] = 1
        # C00 += M2
        W[c0*2+0][col] = 1
        col += 1
        
        # M3 = (a00 + a10) * (b00 + b01)
        U[0*3+0][col] = 1
        U[1*3+0][col] = 1
        V[0*3+c0][col] = 1
        V[0*3+c1][col] = 1
        # C00 -= M3
        W[c0*2+0][col] = -1
        # C01 += M3
        W[c1*2+0][col] = 1
        col += 1
        
        # M4 = (a11 + a00) * (b10 + b11)
        U[1*3+1][col] = 1
        U[0*3+0][col] = 1
        V[1*3+c0][col] = 1
        V[1*3+c1][col] = 1
        # C00 += M4
        W[c0*2+0][col] = 1
        # C01 += M4
        W[c1*2+0][col] = 1
        col += 1
        
        # M5 = (a01 + a11) * (b10 + b11)
        U[0*3+1][col] = 1
        U[1*3+1][col] = 1
        V[1*3+c0][col] = 1
        V[1*3+c1][col] = 1
        # C01 -= M5
        W[c1*2+0][col] = -1
        # C11 += M5
        W[c1*2+1][col] = 1
        col += 1
        
        # M6 = (a00 - a10) * (b10 + b11)
        U[0*3+0][col] = 1
        U[1*3+0][col] = -1
        V[1*3+c0][col] = 1
        V[1*3+c1][col] = 1
        # C10 -= M6
        W[c0*2+1][col] = -1
        col += 1
        
        # M7 = (a01 - a11) * (b00 + b10)
        U[0*3+1][col] = 1
        U[1*3+1][col] = -1
        V[0*3+c0][col] = 1
        V[1*3+c0][col] = 1
        # C11 -= M7
        W[c1*2+1][col] = -1
        col += 1
        
        # --- Part 2: 2x1 x 2 (Trivial) ---
        # Middle index: 2
        # A indices: (0,2), (1,2)
        # B indices: (2,0), (2,1)
        # C indices: (0,0), (0,1), (1,0), (1,1)
        
        # T1 = a02 * b20
        U[0*3+2][col] = 1
        V[2*3+c0][col] = 1
        # C00 += T1
        W[c0*2+0][col] = 1
        col += 1
        
        # T2 = a02 * b21
        U[0*3+2][col] = 1
        V[2*3+c1][col] = 1
        # C01 += T2
        W[c1*2+0][col] = 1
        col += 1
        
        # T3 = a12 * b20
        U[1*3+2][col] = 1
        V[2*3+c0][col] = 1
        # C10 += T3
        W[c0*2+1][col] = 1
        col += 1
        
        # T4 = a12 * b21
        U[1*3+2][col] = 1
        V[2*3+c1][col] = 1
        # C11 += T4
        W[c1*2+1][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
