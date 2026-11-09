import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 using nested Strassen
    # Total rank for 4x4 x 4x8 is 49 * 2 = 98
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 2x2 Strassen factors
    # u, v, w are lists of 7 vectors of length 4
    # Order of entries: 00, 01, 10, 11
    u_inner = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    v_inner = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    w_inner = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 0], # M2
        [0, 1, 0, 0], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 1]  # M7
    ]
    
    # Block-level Strassen factors for 2x2 blocks
    # These are 2x2 matrices of coefficients for the 4 sub-blocks
    # u_block[k][bi][bj]
    u_block = [
        [[1, 0], [0, 1]], # M1: A11 + A22 -> blocks (0,0) and (1,1)
        [[0, 0], [1, 1]], # M2: A21 + A22 -> blocks (1,0) and (1,1)
        [[1, 0], [0, 0]], # M3: A11 -> block (0,0)
        [[0, 0], [0, 1]], # M4: A22 -> block (1,1)
        [[1, 1], [0, 0]], # M5: A11 + A12 -> blocks (0,0) and (0,1)
        [[-1, 0], [1, 0]], # M6: A21 - A11 -> blocks (1,0) and (0,0)
        [[0, 1], [0, -1]]  # M7: A12 - A22 -> blocks (0,1) and (1,1)
    ]
    v_block = [
        [[1, 0], [0, 1]], # M1: B11 + B22
        [[1, 0], [0, 0]], # M2: B11
        [[0, 1], [0, -1]], # M3: B12 - B22
        [[-1, 0], [1, 0]], # M4: B21 - B11
        [[0, 0], [0, 1]], # M5: B22
        [[1, 1], [0, 0]], # M6: B11 + B12
        [[0, 0], [1, 1]]  # M7: B21 + B22
    ]
    w_block = [
        [[1, 0], [0, 1]], # M1: C11 + C22
        [[0, 0], [1, 0]], # M2: C21
        [[0, 1], [0, 0]], # M3: C12
        [[1, 0], [1, 0]], # M4: C11 + C21
        [[-1, 1], [0, 0]], # M5: -C11 + C12
        [[0, 0], [0, 1]], # M6: C22
        [[1, 0], [0, 1]]  # M7: C11 + C22
    ]
    
    # Generate factors for 4x4x4
    # Loop over 2 copies (Left and Right half of B)
    for copy in range(2):
        offset = copy * 49
        for k in range(7): # Block Strassen index
            for l in range(7): # Inner Strassen index
                col = offset + k * 7 + l
                
                # U: 16 rows (4x4 A)
                for i in range(4):
                    for j in range(4):
                        bi = i // 2
                        bj = j // 2
                        ii = i % 2
                        jj = j % 2
                        U[i * 4 + j][col] = u_block[k][bi][bj] * u_inner[l][ii * 2 + jj]
                        
                # V: 32 rows (4x8 B)
                # For Left copy (copy=0), B cols 0-3. For Right (copy=1), B cols 4-7.
                for j in range(4):
                    for k2 in range(4):
                        bi = j // 2
                        bj = k2 // 2
                        ii = j % 2
                        jj = k2 % 2
                        # Map k2 (0-3) to global B col
                        global_k2 = k2 + copy * 4
                        v_idx = j * 8 + global_k2
                        V[v_idx][col] = v_block[k][bi][bj] * v_inner[l][ii * 2 + jj]
                        
                # W: 32 rows (8x4 C)
                # For Left copy (copy=0), C cols 0-3. For Right (copy=1), C cols 4-7.
                for k2 in range(4):
                    for i in range(4):
                        bi = k2 // 2
                        bj = i // 2
                        ii = k2 % 2
                        jj = i % 2
                        # Map k2 (0-3) to global C col
                        global_k2 = k2 + copy * 4
                        W[global_k2 * 4 + i][col] = w_block[k][bi][bj] * w_inner[l][ii * 2 + jj]
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
