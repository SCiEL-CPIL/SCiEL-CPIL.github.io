import time
import sys

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS
    # Rank calculation
    # 4 blocks of size 2 (Strassen, rank 7) + 1 block of size 1 (Schoolbook, rank 4)
    num_blocks_2 = p // 2
    num_blocks_1 = p % 2
    rank = num_blocks_2 * 7 + num_blocks_1 * 4
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen blocks
    for block_idx in range(num_blocks_2):
        k_start = block_idx * 2
        k1 = k_start
        k2 = k_start + 1
        
        # M1: (a11 + a22) * (b1,k1 + b2,k2)
        U[0][col] = 1  # a11
        U[3][col] = 1  # a22
        V[k1][col] = 1      # b1,k1
        V[p + k2][col] = 1  # b2,k2
        W[k1 * n + 0][col] = 1   # C1,k1
        W[k2 * n + 1][col] = 1   # C2,k2
        col += 1
        
        # M2: (a21 + a22) * b1,k1
        U[2][col] = 1  # a21
        U[3][col] = 1  # a22
        V[k1][col] = 1      # b1,k1
        W[k1 * n + 1][col] = 1   # C2,k1
        col += 1
        
        # M3: a11 * (b1,k2 - b2,k2)
        U[0][col] = 1  # a11
        V[p + k2][col] = -1  # b2,k2
        V[k2][col] = 1      # b1,k2
        W[k2 * n + 0][col] = 1   # C1,k2
        W[k2 * n + 1][col] = 1   # C2,k2
        col += 1
        
        # M4: a22 * (b2,k1 - b1,k1)
        U[3][col] = 1  # a22
        V[k1][col] = -1      # b1,k1
        V[p + k1][col] = 1   # b2,k1
        W[k1 * n + 0][col] = 1   # C1,k1
        W[k1 * n + 1][col] = 1   # C2,k1
        col += 1
        
        # M5: (a11 + a12) * b2,k2
        U[0][col] = 1  # a11
        U[1][col] = 1  # a12
        V[p + k2][col] = 1  # b2,k2
        W[k1 * n + 0][col] = -1  # -M5 in C1,k1
        W[k2 * n + 0][col] = 1   # +M5 in C1,k2
        col += 1
        
        # M6: (a21 - a11) * (b1,k1 + b1,k2)
        U[0][col] = -1 # a11
        U[2][col] = 1  # a21
        V[k1][col] = 1      # b1,k1
        V[k2][col] = 1      # b1,k2
        W[k2 * n + 1][col] = 1   # C2,k2
        col += 1
        
        # M7: (a12 - a22) * (b2,k1 + b2,k2)
        U[1][col] = 1  # a12
        U[3][col] = -1 # a22
        V[p + k1][col] = 1   # b2,k1
        V[p + k2][col] = 1   # b2,k2
        W[k1 * n + 0][col] = 1   # C1,k1
        col += 1

    # Schoolbook block for the last column
    if num_blocks_1 > 0:
        k = 2 * num_blocks_2
        
        # a11 * b1,k
        U[0][col] = 1
        V[k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        # a12 * b2,k
        U[1][col] = 1
        V[p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        # a21 * b1,k
        U[2][col] = 1
        V[k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
        
        # a22 * b2,k
        U[3][col] = 1
        V[p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
