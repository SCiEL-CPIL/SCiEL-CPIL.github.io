import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a single term
    def add_term(u_indices, v_indices, w_indices, u_vals, v_vals, w_vals):
        nonlocal col
        for idx, val in zip(u_indices, u_vals):
            if val != 0:
                U[idx, col] += val
        for idx, val in zip(v_indices, v_vals):
            if val != 0:
                V[idx, col] += val
        for idx, val in zip(w_indices, w_vals):
            if val != 0:
                W[idx, col] += val
        col += 1

    # Part 1: Schoolbook for bottom row (1x4 * 4x4)
    # A row 2, B all rows, C row 2
    for j in range(4):
        for k in range(4):
            u_idx = [2 * 4 + j]
            v_idx = [j * 4 + k]
            w_idx = [k * 3 + 2]
            add_term(u_idx, v_idx, w_idx, [1], [1], [1])
    
    # Part 2: Strassen for top 2x4 * 4x4
    # Blocks:
    # A_L: rows 0,1 cols 0,1 -> U indices: 0,1,4,5
    # A_R: rows 0,1 cols 2,3 -> U indices: 2,3,6,7
    # B_TL: rows 0,1 cols 0,1 -> V indices: 0,1,4,5
    # B_TR: rows 0,1 cols 2,3 -> V indices: 2,3,6,7
    # B_BL: rows 2,3 cols 0,1 -> V indices: 8,9,12,13
    # B_BR: rows 2,3 cols 2,3 -> V indices: 10,11,14,15
    
    # C_TL: rows 0,1 cols 0,1 -> W indices: 0,3,1,4
    # C_TR: rows 0,1 cols 2,3 -> W indices: 6,9,7,10
    # Note: W index is k*n + i.
    # C[0,0] -> 0*3+0=0
    # C[0,1] -> 1*3+0=3
    # C[1,0] -> 0*3+1=1
    # C[1,1] -> 1*3+1=4
    # C[0,2] -> 2*3+0=6
    # C[0,3] -> 3*3+0=9
    # C[1,2] -> 2*3+1=7
    # C[1,3] -> 3*3+1=10
    
    w_TL = [0, 3, 1, 4]
    w_TR = [6, 9, 7, 10]
    
    a_L = [0, 1, 4, 5]
    a_R = [2, 3, 6, 7]
    
    b_TL = [0, 1, 4, 5]
    b_TR = [2, 3, 6, 7]
    b_BL = [8, 9, 12, 13]
    b_BR = [10, 11, 14, 15]
    
    def add_strassen_block(a_idx, b_idx, w_idx):
        nonlocal col
        
        a11, a12, a21, a22 = a_idx
        b11, b12, b21, b22 = b_idx
        c11, c12, c21, c22 = w_idx
        
        # M1 = (a11 + a22)(b11 + b22)
        add_term([a11, a22], [b11, b22], [c11, c22], [1, 1], [1, 1], [1, 1])
        
        # M2 = (a21 + a22)b11
        add_term([a21, a22], [b11], [c21, c22], [1, 1], [1], [1, -1])
        
        # M3 = a11(b21 - b22)
        add_term([a11], [b21, b22], [c12, c22], [1], [1, -1], [1, 1])
        
        # M4 = a22(b11 - b21)
        add_term([a22], [b11, b21], [c11, c21], [1], [1, -1], [1, 1])
        
        # M5 = (a11 + a12)b22
        add_term([a11, a12], [b22], [c11, c12], [1, 1], [1], [-1, 1])
        
        # M6 = (a21 - a11)(b11 + b12)
        add_term([a21, a11], [b11, b12], [c22], [1, -1], [1, 1], [1])
        
        # M7 = (a12 - a22)(b21 + b22)
        add_term([a12, a22], [b21, b22], [c11], [1, -1], [1, 1], [1])
        
    # Add blocks
    add_strassen_block(a_L, b_TL, w_TL)
    add_strassen_block(a_R, b_BL, w_TL)
    add_strassen_block(a_L, b_TR, w_TR)
    add_strassen_block(a_R, b_BR, w_TR)
    
    return {"U": U, "V": V, "W": W}
