"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # 20s safety margin
    
    # Build a structured decomposition using 2x2 Strassen blocks
    # A (4x4) = [[A11, A12], [A21, A22]] where each is 2x2
    # B (4x8) split into left (4x4) and right (4x4)
    # C (4x8) split into left (4x4) and right (4x4)
    #
    # For (4x4)*(4x4) using 2x2 Strassen: rank 28 (7 Strassen terms * 4 scalar mults each)
    # For (4x4)*(4x8): 2 * 28 = 56
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 16 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 32 x 56
    
    col = 0
    
    # Strassen M-j coefficients for A (2x2 block local indices 0-3)
    A_coeffs = {
        0: {0: 1, 3: 1},  # M1: A11, A22
        1: {2: 1, 3: 1},  # M2: A21, A22
        2: {0: 1},         # M3: A11
        3: {3: 1},         # M4: A22
        4: {0: 1, 1: 1},  # M5: A11, A12
        5: {2: 1, 0: -1}, # M6: A21, -A11
        6: {1: 1, 3: -1}, # M7: A12, -A22
    }
    
    # Strassen M-j coefficients for B (2x2 block local indices 0-3)
    B_coeffs = {
        0: {0: 1, 3: 1},  # M1: B11, B22
        1: {0: 1},         # M2: B11
        2: {1: 1, 3: -1}, # M3: B12, -B22
        3: {2: 1, 0: -1}, # M4: B21, -B11
        4: {3: 1},         # M5: B22
        5: {0: 1, 1: 1},  # M6: B11, B12
        6: {2: 1, 3: 1},  # M7: B21, B22
    }
    
    # C output coefficients: for each C block (0=C11, 1=C12, 2=C21, 3=C22)
    C_coeffs = {
        0: {0: 1, 3: -1, 4: 1, 6: -1},  # C11: M1, -M4, M5, -M7
        1: {2: 1, 6: 1},                 # C12: M3, M7
        2: {1: 1, 3: 1},                 # C21: M2, M4
        3: {0: 1, 2: -1, 5: 1, 6: 1},   # C22: M1, -M3, M6, M7
    }
    
    # Local index to (row_block, col_block) for 2x2 blocks
    block_map = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
    
    # Process left half (B cols 0-3, C cols 0-3)
    for mj in range(7):
        # A part
        for local_idx, coeff in A_coeffs[mj].items():
            a_rb, a_cb = block_map[local_idx]
            lr = local_idx // 2
            lc = local_idx % 2
            global_row = a_rb * 2 + lr
            global_col = a_cb * 2 + lc
            U[global_row * m + global_col][col] = coeff
        
        # B part (left half)
        for local_idx, coeff in B_coeffs[mj].items():
            b_rb, b_cb = block_map[local_idx]
            lr = local_idx // 2
            lc = local_idx % 2
            global_row = b_rb * 2 + lr
            global_col = b_cb * 2 + lc
            V[global_row * p + global_col][col] = coeff
        
        # C part (left half)
        for c_block, c_coeff_dict in C_coeffs.items():
            if mj in c_coeff_dict:
                c_rb, c_cb = block_map[c_block]
                for lr in range(2):
                    for lc in range(2):
                        global_row = c_rb * 2 + lr
                        global_col = c_cb * 2 + lc
                        W[global_col * n + global_row][col] += c_coeff_dict[mj]
        
        col += 1
    
    # Process right half (B cols 4-7, C cols 4-7)
    for mj in range(7):
        # A part (same)
        for local_idx, coeff in A_coeffs[mj].items():
            a_rb, a_cb = block_map[local_idx]
            lr = local_idx // 2
            lc = local_idx % 2
            global_row = a_rb * 2 + lr
            global_col = a_cb * 2 + lc
            U[global_row * m + global_col][col] = coeff
        
        # B part (right half: cols offset by 4)
        for local_idx, coeff in B_coeffs[mj].items():
            b_rb, b_cb = block_map[local_idx]
            lr = local_idx // 2
            lc = local_idx % 2
            global_row = b_rb * 2 + lr
            global_col = 4 + b_cb * 2 + lc
            V[global_row * p + global_col][col] = coeff
        
        # C part (right half: cols offset by 4)
        for c_block, c_coeff_dict in C_coeffs.items():
            if mj in c_coeff_dict:
                c_rb, c_cb = block_map[c_block]
                for lr in range(2):
                    for lc in range(2):
                        global_row = c_rb * 2 + lr
                        global_col = 4 + c_cb * 2 + lc
                        W[global_col * n + global_row][col] += c_coeff_dict[mj]
        
        col += 1
    
    # Verify the decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    correct = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                if abs(val - 1.0) > 1e-10:
                    correct = False
                    break
            if not correct:
                break
        if not correct:
            break
    
    if not correct:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
