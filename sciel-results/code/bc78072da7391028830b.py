import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using two independent Strassen 2x2x2 multiplications for each half of the output?
    # Actually, we split B into two 2x2 blocks for the left half of C, and two for the right half.
    # C_left = A_left * B_top_left + A_right * B_bot_left
    # C_right = A_left * B_top_right + A_right * B_bot_right
    
    # We compute 4 products of 2x2 matrices.
    # P1 = A_left * B_top_left
    # P2 = A_right * B_bot_left
    # P3 = A_left * B_top_right
    # P4 = A_right * B_bot_right
    
    # Each 2x2 product uses Strassen (rank 7).
    # Total rank 28.
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)] # 8 x 28
    V = [[0] * rank for _ in range(m * p)] # 16 x 28
    W = [[0] * rank for _ in range(p * n)] # 8 x 28
    
    # Indices for A (2x4):
    # A[0,0], A[0,1], A[0,2], A[0,3] -> 0, 1, 2, 3
    # A[1,0], A[1,1], A[1,2], A[1,3] -> 4, 5, 6, 7
    
    # Indices for B (4x4):
    # B[0,0]..B[0,3] -> 0..3
    # B[1,0]..B[1,3] -> 4..7
    # B[2,0]..B[2,3] -> 8..11
    # B[3,0]..B[3,3] -> 12..15
    
    # Indices for C (2x4):
    # C[0,0]..C[0,3] -> 0..3
    # C[1,0]..C[1,3] -> 4..7
    
    # Helper to add a Strassen term for a 2x2 product
    # A_indices: [a00, a01, a10, a11]
    # B_indices: [b00, b01, b10, b11]
    # C_indices: [c00, c01, c10, c11]
    # base_col: starting column in U, V, W
    
    def add_strassen(A_idx, B_idx, C_idx, base_col):
        # M1 = (a00 + a11) * (b00 + b11)
        U[A_idx[0]][base_col] += 1
        U[A_idx[3]][base_col] += 1
        V[B_idx[0]][base_col] += 1
        V[B_idx[3]][base_col] += 1
        W[C_idx[0]][base_col] += 1
        W[C_idx[3]][base_col] += 1
        W[C_idx[2]][base_col] -= 1
        W[C_idx[1]][base_col] -= 1
        
        # M2 = (a10 + a11) * b00
        U[A_idx[2]][base_col+1] += 1
        U[A_idx[3]][base_col+1] += 1
        V[B_idx[0]][base_col+1] += 1
        W[C_idx[3]][base_col+1] += 1
        W[C_idx[2]][base_col+1] -= 1
        
        # M3 = a00 * (b01 - b11)
        U[A_idx[0]][base_col+2] += 1
        V[B_idx[1]][base_col+2] += 1
        V[B_idx[3]][base_col+2] -= 1
        W[C_idx[1]][base_col+2] += 1
        W[C_idx[3]][base_col+2] -= 1
        
        # M4 = a11 * (b10 - b00)
        U[A_idx[3]][base_col+3] += 1
        V[B_idx[2]][base_col+3] += 1
        V[B_idx[0]][base_col+3] -= 1
        W[C_idx[2]][base_col+3] += 1
        W[C_idx[0]][base_col+3] -= 1
        
        # M5 = (a00 + a01) * b11
        U[A_idx[0]][base_col+4] += 1
        U[A_idx[1]][base_col+4] += 1
        V[B_idx[3]][base_col+4] += 1
        W[C_idx[1]][base_col+4] += 1
        W[C_idx[0]][base_col+4] -= 1
        
        # M6 = (a10 - a00) * (b00 + b01)
        U[A_idx[2]][base_col+5] += 1
        U[A_idx[0]][base_col+5] -= 1
        V[B_idx[0]][base_col+5] += 1
        V[B_idx[1]][base_col+5] += 1
        W[C_idx[2]][base_col+5] += 1
        W[C_idx[0]][base_col+5] -= 1
        
        # M7 = (a01 - a11) * (b10 + b11)
        U[A_idx[1]][base_col+6] += 1
        U[A_idx[3]][base_col+6] -= 1
        V[B_idx[2]][base_col+6] += 1
        V[B_idx[3]][base_col+6] += 1
        W[C_idx[1]][base_col+6] += 1
        W[C_idx[3]][base_col+6] -= 1
        
    # A_left: rows 0,1 cols 0,1 -> [0, 1, 4, 5]
    # A_right: rows 0,1 cols 2,3 -> [2, 3, 6, 7]
    
    # B_top_left: rows 0,1 cols 0,1 -> [0, 1, 4, 5]
    # B_bot_left: rows 2,3 cols 0,1 -> [8, 9, 12, 13]
    # B_top_right: rows 0,1 cols 2,3 -> [2, 3, 6, 7]
    # B_bot_right: rows 2,3 cols 2,3 -> [10, 11, 14, 15]
    
    # C_left: rows 0,1 cols 0,1 -> [0, 1, 4, 5]
    # C_right: rows 0,1 cols 2,3 -> [2, 3, 6, 7]
    
    A_left = [0, 1, 4, 5]
    A_right = [2, 3, 6, 7]
    
    B_TL = [0, 1, 4, 5]
    B_BL = [8, 9, 12, 13]
    B_TR = [2, 3, 6, 7]
    B_BR = [10, 11, 14, 15]
    
    C_L = [0, 1, 4, 5]
    C_R = [2, 3, 6, 7]
    
    # P1 = A_left * B_TL contributes to C_L
    add_strassen(A_left, B_TL, C_L, 0)
    
    # P2 = A_right * B_BL contributes to C_L
    add_strassen(A_right, B_BL, C_L, 7)
    
    # P3 = A_left * B_TR contributes to C_R
    add_strassen(A_left, B_TR, C_R, 14)
    
    # P4 = A_right * B_BR contributes to C_R
    add_strassen(A_right, B_BR, C_R, 21)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
