import numpy as np
import sys
import signal
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = start_time + 110.0

    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 2: A_bot * B_right
    # A_bot is A[:, 2] (3x1)
    # B_right is B[2, :] (1x8)
    # C[i, k] += A[i, 2] * B[2, k]
    for i in range(3):
        for k in range(8):
            U[i * 3 + 2][col] = 1
            V[2 * 8 + k][col] = 1
            W[k * 3 + i][col] = 1
            col += 1
            
    # Part 1: A_top * B_left
    # A_top is A[:, 0:2] (3x2)
    # B_left is B[0:2, :] (2x8)
    # Split B_left into 4 blocks of 2x2
    for block_idx in range(4):
        # Columns in B: 4*block_idx, 4*block_idx + 1
        k0 = 4 * block_idx
        k1 = 4 * block_idx + 1
        
        # b11 = B[0, k0], b12 = B[0, k1]
        # b21 = B[1, k0], b22 = B[1, k1]
        
        # Strassen for A[0:2, 0:2] * B[0:2, k0:k1+1]
        # M1 = (A[0,0] + A[1,1]) * (B[0, k0] + B[1, k1])
        U[0 * 3 + 0][col] = 1
        U[1 * 3 + 1][col] = 1
        V[0 * 8 + k0][col] = 1
        V[1 * 8 + k1][col] = 1
        W[k0 * 3 + 0][col] = 1
        W[k1 * 3 + 0][col] = 1
        W[k0 * 3 + 1][col] = 1
        W[k1 * 3 + 1][col] = 1
        col += 1
        
        # M2 = (A[1,0] + A[1,1]) * B[0, k0]
        U[1 * 3 + 0][col] = 1
        U[1 * 3 + 1][col] = 1
        V[0 * 8 + k0][col] = 1
        W[k0 * 3 + 1][col] = 1
        col += 1
        
        # M3 = A[0,0] * (B[0, k1] - B[1, k1])
        U[0 * 3 + 0][col] = 1
        V[0 * 8 + k1][col] = 1
        V[1 * 8 + k1][col] = -1
        W[k1 * 3 + 0][col] = 1
        col += 1
        
        # M4 = A[1,1] * (B[1, k0] - B[0, k0])
        U[1 * 3 + 1][col] = 1
        V[1 * 8 + k0][col] = 1
        V[0 * 8 + k0][col] = -1
        W[k0 * 3 + 1][col] = 1
        col += 1
        
        # M5 = (A[0,0] + A[0,1]) * B[1, k1]
        U[0 * 3 + 0][col] = 1
        U[0 * 3 + 1][col] = 1
        V[1 * 8 + k1][col] = 1
        W[k1 * 3 + 0][col] = 1
        col += 1
        
        # M6 = (A[1,0] - A[0,0]) * (B[0, k0] + B[0, k1])
        U[1 * 3 + 0][col] = 1
        U[0 * 3 + 0][col] = -1
        V[0 * 8 + k0][col] = 1
        V[0 * 8 + k1][col] = 1
        W[k0 * 3 + 1][col] = 1
        W[k1 * 3 + 1][col] = 1
        col += 1
        
        # M7 = (A[0,1] - A[1,1]) * (B[1, k0] + B[1, k1])
        U[0 * 3 + 1][col] = 1
        U[1 * 3 + 1][col] = -1
        V[1 * 8 + k0][col] = 1
        V[1 * 8 + k1][col] = 1
        W[k0 * 3 + 1][col] = 1
        W[k1 * 3 + 1][col] = 1
        col += 1
        
        # Schoolbook for A[2, 0:2] * B[0:2, k0:k1+1]
        # C[2, k0] += A[2,0] * B[0, k0] + A[2,1] * B[1, k0]
        # Term 1: A[2,0] * B[0, k0]
        U[2 * 3 + 0][col] = 1
        V[0 * 8 + k0][col] = 1
        W[k0 * 3 + 2][col] = 1
        col += 1
        
        # Term 2: A[2,1] * B[1, k0]
        U[2 * 3 + 1][col] = 1
        V[1 * 8 + k0][col] = 1
        W[k0 * 3 + 2][col] = 1
        col += 1
        
        # Term 3: A[2,0] * B[0, k1]
        U[2 * 3 + 0][col] = 1
        V[0 * 8 + k1][col] = 1
        W[k1 * 3 + 2][col] = 1
        col += 1
        
        # Term 4: A[2,1] * B[1, k1]
        U[2 * 3 + 1][col] = 1
        V[1 * 8 + k1][col] = 1
        W[k1 * 3 + 2][col] = 1
        col += 1
        
    if col != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {col}")

    U_result = [[int(x) if abs(x - round(x)) < 1e-9 else round(x, 6) for x in row] for row in U]
    V_result = [[int(x) if abs(x - round(x)) < 1e-9 else round(x, 6) for x in row] for row in V]
    W_result = [[int(x) if abs(x - round(x)) < 1e-9 else round(x, 6) for x in row] for row in W]

    return {"U": U_result, "V": V_result, "W": W_result}

# EVOLVE-BLOCK-END
