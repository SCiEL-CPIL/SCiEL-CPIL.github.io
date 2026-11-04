"""Rank-minimal tensor decomposition for 2x3 * 3x10 matrix multiplication.
Uses Strassen's algorithm for 2x2 blocks and schoolbook for the remainder.
Known rank: 55.
"""
import numpy as np
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-55 decomposition for 2x3 * 3x10 matrix multiplication.
    U: (6, 55), V: (30, 55), W: (20, 55)
    """
    n, m, p = 2, 3, 10
    rank = 55
    
    # Initialize factor matrices
    U_mat = [[0] * rank for _ in range(n * m)] # 6 x 55
    V_mat = [[0] * rank for _ in range(m * p)] # 30 x 55
    W_mat = [[0] * rank for _ in range(p * n)] # 20 x 55
    
    idx = 0
    
    # Strategy:
    # Split B (3x10) into B_top (2x10) and B_bottom (1x10).
    # Split A (2x3) into A_left (2x2) and A_right (2x1).
    # C = A_left @ B_top + A_right @ B_bottom
    
    # Part 1: A_left (2x2) @ B_top (2x10)
    # B_top has 10 columns. We process them in pairs (k0, k1) for Strassen.
    # 5 pairs of columns.
    for l in range(5):
        k0 = 2 * l
        k1 = 2 * l + 1
        
        # A indices (row, col) -> row*m + col
        a00 = 0 * 3 + 0 # 0
        a01 = 0 * 3 + 1 # 1
        a10 = 1 * 3 + 0 # 3
        a11 = 1 * 3 + 1 # 4
        
        # B indices (row, col) -> row*p + col
        # B_top rows are 0, 1.
        b0k0 = 0 * 10 + k0
        b0k1 = 0 * 10 + k1
        b1k0 = 1 * 10 + k0
        b1k1 = 1 * 10 + k1
        
        # C indices (row, col) -> col*n + row
        # C rows are 0, 1. C cols are k0, k1.
        c0k0 = k0 * 2 + 0
        c0k1 = k1 * 2 + 0
        c1k0 = k0 * 2 + 1
        c1k1 = k1 * 2 + 1
        
        # Strassen 7 multiplications for 2x2 @ 2x2 -> 2x2
        
        # M1 = (a00 + a11) * (b0k0 + b1k1)
        U_mat[a00][idx] += 1
        U_mat[a11][idx] += 1
        V_mat[b0k0][idx] += 1
        V_mat[b1k1][idx] += 1
        W_mat[c0k0][idx] += 1
        W_mat[c1k1][idx] += 1
        idx += 1
        
        # M2 = (a10 + a11) * b0k0
        U_mat[a10][idx] += 1
        U_mat[a11][idx] += 1
        V_mat[b0k0][idx] += 1
        W_mat[c1k0][idx] += 1
        W_mat[c1k1][idx] -= 1
        idx += 1
        
        # M3 = a00 * (b0k1 - b1k1)
        U_mat[a00][idx] += 1
        V_mat[b0k1][idx] += 1
        V_mat[b1k1][idx] -= 1
        W_mat[c0k1][idx] += 1
        W_mat[c1k1][idx] += 1
        idx += 1
        
        # M4 = a11 * (b1k0 - b0k0)
        U_mat[a11][idx] += 1
        V_mat[b1k0][idx] += 1
        V_mat[b0k0][idx] -= 1
        W_mat[c0k0][idx] += 1
        W_mat[c1k0][idx] += 1
        idx += 1
        
        # M5 = (a00 + a01) * b1k1
        U_mat[a00][idx] += 1
        U_mat[a01][idx] += 1
        V_mat[b1k1][idx] += 1
        W_mat[c0k0][idx] -= 1
        W_mat[c0k1][idx] += 1
        idx += 1
        
        # M6 = (a10 - a00) * (b0k0 + b0k1)
        U_mat[a10][idx] += 1
        U_mat[a00][idx] -= 1
        V_mat[b0k0][idx] += 1
        V_mat[b0k1][idx] += 1
        W_mat[c1k1][idx] += 1
        idx += 1
        
        # M7 = (a01 - a11) * (b1k0 + b1k1)
        U_mat[a01][idx] += 1
        U_mat[a11][idx] -= 1
        V_mat[b1k0][idx] += 1
        V_mat[b1k1][idx] += 1
        W_mat[c0k0][idx] += 1
        idx += 1

    # Part 2: A_right (2x1) @ B_bottom (1x10)
    # A_right is column 2 of A. Indices: 0*3+2=2, 1*3+2=5.
    # B_bottom is row 2 of B. Indices: 2*10+k.
    # Result adds to C.
    for i in range(2):
        for k in range(10):
            u_idx = i * 3 + 2
            v_idx = 2 * 10 + k
            w_idx = k * 2 + i
            
            U_mat[u_idx][idx] += 1
            V_mat[v_idx][idx] += 1
            W_mat[w_idx][idx] += 1
            idx += 1
            
    return {'U': U_mat, 'V': V_mat, 'W': W_mat}
# EVOLVE-BLOCK-END
