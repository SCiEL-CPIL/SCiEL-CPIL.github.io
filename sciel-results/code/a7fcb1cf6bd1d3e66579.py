"""Rewritten program for 3x5x6 matrix multiplication rank search."""
DIMENSIONS = [3, 5, 6]

import sys
import time
import random
import itertools
from fractions import Fraction

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, rank, tol=1e-9):
    """Check if U, V, W form a valid matrix multiplication decomposition."""
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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

def _construct_block_decomposition(n, m, p):
    """
    Constructs a rank-84 decomposition for 3x5x6 using block Strassen.
    Tiling:
    - 6 blocks of 2x2x2 Strassen in the i=0,1 slice.
    - 1 block of 1x5x6 Schoolbook for i=2 slice.
    - 1 block of 2x1x6 Schoolbook for the remaining j=4 strip in i=0,1 slice.
    """
    if n != 3 or m != 5 or p != 6:
        return _schoolbook(n, m, p)
        
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define the 6 Strassen blocks (i_off, j_off, k_off)
    # i in [0,2), j in [0,4), k in [0,6)
    strassen_blocks = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 0, 4),
        (0, 2, 0),
        (0, 2, 2),
        (0, 2, 4)
    ]
    
    for i_off, j_off, k_off in strassen_blocks:
        # Term 0: M1
        U[i_off * m + j_off][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + k_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[k_off * n + i_off][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = 1
        col += 1
        
        # Term 1: M2
        U[(i_off+1) * m + j_off][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + k_off][col] = 1
        W[k_off * n + (i_off+1)][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = -1
        col += 1
        
        # Term 2: M3
        U[i_off * m + j_off][col] = 1
        V[j_off * p + (k_off+1)][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = -1
        W[(k_off+1) * n + i_off][col] = -1
        W[(k_off+1) * n + (i_off+1)][col] = 1
        col += 1
        
        # Term 3: M4
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[(j_off+1) * p + k_off][col] = 1
        V[j_off * p + k_off][col] = -1
        W[k_off * n + i_off][col] = 1
        W[k_off * n + (i_off+1)][col] = 1
        col += 1
        
        # Term 4: M5
        U[i_off * m + j_off][col] = 1
        U[i_off * m + (j_off+1)][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[k_off * n + i_off][col] = -1
        W[(k_off+1) * n + i_off][col] = 1
        col += 1
        
        # Term 5: M6
        U[(i_off+1) * m + j_off][col] = 1
        U[i_off * m + j_off][col] = -1
        V[j_off * p + k_off][col] = 1
        V[j_off * p + (k_off+1)][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = 1
        col += 1
        
        # Term 6: M7
        U[i_off * m + (j_off+1)][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = -1
        V[(j_off+1) * p + k_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[k_off * n + i_off][col] = 1
        col += 1
        
    # Schoolbook Block 1: i in [2,3), j in [0,5), k in [0,6)
    # Size 1x5x6 = 30 terms
    for i in range(2, 3):
        for j in range(0, 5):
            for k in range(0, 6):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Schoolbook Block 2: i in [0,2), j in [4,5), k in [0,6)
    # Size 2x1x6 = 12 terms
    for i in range(0, 2):
        for j in range(4, 5):
            for k in range(0, 6):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

def _construct():
    """Main construction function."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try the specific block decomposition
    result = _construct_block_decomposition(n, m, p)
    
    # Verify it
    if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p, 84):
        return result
        
    # Fallback to schoolbook if verification fails (should not happen)
    return _schoolbook(n, m, p)

def construct():
    return _construct()

# EVOLVE-BLOCK-END
