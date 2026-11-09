import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We want to multiply 4x4 by 4x8.
    # This is equivalent to two 4x4 by 4x4 multiplications.
    # We use rank 49 for 4x4x4 (Strassen nested).
    # Total rank = 49 * 2 = 98.
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 2x2 Strassen factors
    # u, v, w are lists of 7 vectors of length 4
    # Order of entries: 00, 01, 10, 11
    u_2x2 = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    v_2x2 = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    w_2x2 = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 0], # M2
        [0, 1, 0, 0], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 1]  # M7
    ]
    
    # Check 2x2 decomposition
    # C_11 = M1 + M4 - M5 + M7
    # C_12 = M3 + M5
    # C_21 = M2 + M4
    # C_22 = M1 - M2 + M3 + M6
    
    # Super-product factors (for 4x4 blocks)
    # u_super[k] is a vector of length 4, coeff for blocks A0, A1, A2, A3
    u_super = [
        [1, 0, 0, 1], # M1: A11 + A22
        [0, 0, 1, 1], # M2: A21 + A22
        [1, 0, 0, 0], # M3: A11
        [0, 0, 0, 1], # M4: A22
        [1, 1, 0, 0], # M5: A11 + A12
        [-1, 0, 1, 0], # M6: A21 - A11
        [0, 1, 0, -1]  # M7: A12 - A22
    ]
    v_super = [
        [1, 0, 0, 1], # M1: B11 + B22
        [1, 0, 0, 0], # M2: B11
        [0, 1, 0, -1], # M3: B12 - B22
        [-1, 0, 1, 0], # M4: B21 - B11
        [0, 0, 0, 1], # M5: B22
        [1, 1, 0, 0], # M6: B11 + B12
        [0, 0, 1, 1]  # M7: B21 + B22
    ]
    w_super = [
        [1, 0, 0, 1], # M1: C11 + C22
        [0, 0, 1, 0], # M2: C21
        [0, 1, 0, 0], # M3: C12
        [1, 0, 1, 0], # M4: C11 + C21
        [-1, 1, 0, 0], # M5: -C11 + C12
        [0, 0, 0, 1], # M6: C22
        [1, 0, 0, 1]  # M7: C11 + C22
    ]
    
    # Construct 4x4 factors
    # For each of the 2 copies (left and right half of B)
    # Left half: columns 0-3 of B
    # Right half: columns 4-7 of B
    
    for copy in range(2):
        offset = copy * 49
        for k in range(7): # Super product index
            for l in range(7): # Inner product index
                col = offset + k * 7 + l
                # U
                for i in range(4):
                    for j in range(4):
                        bi = i // 2
                        bj = j // 2
                        ii = i % 2
                        jj = j % 2
                        block_idx = bi * 2 + bj
                        U[i * 4 + j][col] = u_super[k][block_idx] * u_2x2[l][ii * 2 + jj]
                        
                # V
                for j in range(4):
                    for k2 in range(4):
                        bi = j // 2
                        bj = k2 // 2
                        ii = j % 2
                        jj = k2 % 2
                        block_idx = bi * 2 + bj
                        # V index: j * 8 + k2 + copy * 4
                        v_idx = j * 8 + (k2 + copy * 4)
                        V[v_idx][col] = v_super[k][block_idx] * v_2x2[l][ii * 2 + jj]
                        
                # W
                for k2 in range(4):
                    for i in range(4):
                        bi = k2 // 2
                        bj = i // 2
                        ii = k2 % 2
                        jj = i % 2
                        block_idx = bi * 2 + bj
                        # W index: k2 * 4 + i
                        W[k2 * 4 + i][col] = w_super[k][block_idx] * w_2x2[l][ii * 2 + jj]
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
