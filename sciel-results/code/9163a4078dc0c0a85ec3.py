"""Matrix multiplication 2,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]
START_TIME = time.time()
DEADLINE = START_TIME + 120  # 120s wall-clock limit

def _time_remaining():
    return DEADLINE - time.time()

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    
    # We use the known rank 54 decomposition based on Strassen for 2x2 blocks
    # and schoolbook for the remaining 1x1 or 2x1 parts.
    # Decomposition:
    # A = [A1 (2x2), A2 (2x2), A3 (2x1)]
    # B = [B1 (2x6), B2 (2x6), B3 (1x6)]
    # C = A1 B1 + A2 B2 + A3 B3
    
    # A1 B1: 2x2 * 2x6. Split B1 into three 2x2 blocks.
    # Each 2x2 * 2x2 is Strassen (rank 7). Total 3 * 7 = 21.
    # A2 B2: 2x2 * 2x6. Split B2 into three 2x2 blocks.
    # Each 2x2 * 2x2 is Strassen (rank 7). Total 3 * 7 = 21.
    # A3 B3: 2x1 * 1x6. Split B3 into three 1x2 blocks? No, 2x1 * 1x2 is rank 4.
    # 3 * 4 = 12.
    # Total rank = 21 + 21 + 12 = 54.

    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base):
        nonlocal col
        # Indices: U[i*m + j], V[j*p + k], W[k*n + i]
        # Block is 2x2 * 2x2 -> 2x2
        # i_base, j_base, k_base are the top-left indices
        
        # M0: (A00+A11)(B00+B11)
        U[i_base * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # M1: (A10+A11)B00
        U[(i_base+1) * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] -= 1
        col += 1
        
        # M2: A00(B01-B11)
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # M3: A11(B10-B00)
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + k_base][col] += 1
        V[j_base * p + k_base][col] -= 1
        W[k_base * n + i_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1
        
        # M4: (A00+A01)B11
        U[i_base * m + j_base][col] += 1
        U[i_base * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1
        
        # M5: (A10-A00)(B00+B01)
        U[(i_base+1) * m + j_base][col] += 1
        U[i_base * m + j_base][col] -= 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # M6: (A01-A11)(B10+B11)
        U[i_base * m + (j_base+1)][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] -= 1
        V[(j_base+1) * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1

    def add_2x1_1x2_block(i_base, j_base, k_base):
        nonlocal col
        # 2x1 * 1x2 -> 2x2
        # A is 2x1 (rows i_base, i_base+1, col j_base)
        # B is 1x2 (row j_base, cols k_base, k_base+1)
        # C is 2x2
        
        # C00 = A00 * B00
        U[i_base * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1
        
        # C01 = A00 * B01
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1
        
        # C10 = A10 * B00
        U[(i_base+1) * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1
        
        # C11 = A10 * B01
        U[(i_base+1) * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1

    # A1 is rows 0,1 cols 0,1. B1 is rows 0,1 cols 0,5.
    # We split B1 into cols (0,1), (2,3), (4,5).
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    
    # A2 is rows 0,1 cols 2,3. B2 is rows 2,3 cols 0,5.
    # We split B2 into cols (0,1), (2,3), (4,5).
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(0, 2, 4)
    
    # A3 is rows 0,1 col 4. B3 is row 4 cols 0,5.
    # We split B3 into cols (0,1), (2,3), (4,5).
    add_2x1_1x2_block(0, 4, 0)
    add_2x1_1x2_block(0, 4, 2)
    add_2x1_1x2_block(0, 4, 4)
    
    # Trim to actual rank used
    if col < rank:
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
        rank = col

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
