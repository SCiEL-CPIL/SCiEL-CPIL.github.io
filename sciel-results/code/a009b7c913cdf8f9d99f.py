import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    # Strassen vectors for 2x2 matrix multiplication
    # u, v, w are lists of 7 vectors, each of length 4 (flattened 2x2)
    # Indices: 00, 01, 10, 11
    U_2x2 = [
        [1, 0, 0, 1], # M1: X11 + X22
        [0, 0, 1, 1], # M2: X21 + X22
        [1, 0, 0, 0], # M3: X11
        [0, 0, 0, 1], # M4: X22
        [1, 1, 0, 0], # M5: X11 + X12
        [-1, 0, 1, 0],# M6: X21 - X11
        [0, 1, 0, -1] # M7: X12 - X22
    ]
    
    V_2x2 = [
        [1, 0, 0, 1], # M1: Y11 + Y22
        [1, 0, 0, 0], # M2: Y11
        [0, 1, 0, -1],# M3: Y12 - Y22
        [-1, 0, 1, 0],# M4: Y21 - Y11
        [0, 0, 0, 1], # M5: Y22
        [1, 1, 0, 0], # M6: Y11 + Y12
        [0, 0, 1, 1]  # M7: Y21 + Y22
    ]
    
    W_2x2 = [
        [1, 0, 0, 1], # M1: Z11 + Z22
        [0, 0, 1, -1],# M2: Z21 - Z22
        [0, 1, 0, 1], # M3: Z12 + Z22
        [1, 0, 1, 0], # M4: Z11 + Z21
        [-1, 1, 0, 0],# M5: -Z11 + Z12
        [0, 0, 0, 1], # M6: Z22
        [1, 0, 0, 0]  # M7: Z11
    ]
    
    # Outer Strassen coefficients for 4x4 (blocks)
    # u, v, w are lists of 7 vectors, each of length 4 (flattened 2x2 blocks)
    # Block indices: 00, 01, 10, 11
    U_4x4 = [
        [1, 0, 0, 1], # M1: A11 + A22
        [0, 0, 1, 1], # M2: A21 + A22
        [1, 0, 0, 0], # M3: A11
        [0, 0, 0, 1], # M4: A22
        [1, 1, 0, 0], # M5: A11 + A12
        [-1, 0, 1, 0],# M6: A21 - A11
        [0, 1, 0, -1] # M7: A12 - A22
    ]
    
    V_4x4 = [
        [1, 0, 0, 1], # M1: B11 + B22
        [1, 0, 0, 0], # M2: B11
        [0, 1, 0, -1],# M3: B12 - B22
        [-1, 0, 1, 0],# M4: B21 - B11
        [0, 0, 0, 1], # M5: B22
        [1, 1, 0, 0], # M6: B11 + B12
        [0, 0, 1, 1]  # M7: B21 + B22
    ]
    
    W_4x4 = [
        [1, 0, 0, 1], # M1: C11 + C22
        [0, 0, 1, -1],# M2: C21 - C22
        [0, 1, 0, 1], # M3: C12 + C22
        [1, 0, 1, 0], # M4: C11 + C21
        [-1, 1, 0, 0],# M5: -C11 + C12
        [0, 0, 0, 1], # M6: C22
        [1, 0, 0, 0]  # M7: C11
    ]
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to map flat index to block and inner index
    # For 4x4, index 0-15
    # Block 00: 0-3, Block 01: 4-7, Block 10: 8-11, Block 11: 12-15
    # Inner index 0-3 corresponds to 00, 01, 10, 11 within the 2x2 block
    
    # Construct U
    for a in range(n * m):
        block_idx = a // 4
        inner_idx = a % 4
        # block_idx 0: 00, 1: 01, 2: 10, 3: 11
        # inner_idx 0: 00, 1: 01, 2: 10, 3: 11
        
        for k in range(7):
            # Coefficient of A_block in X_k
            # X_k is a 2x2 matrix.
            # X_k[inner_idx] gets contribution from A_block if U_4x4[k][block_idx] != 0
            # But U_4x4[k] is a vector of 4 block coeffs.
            # The inner structure is the same for all blocks in X_k.
            # So X_k[inner_idx] = sum_{b} U_4x4[k][b] * A_b[inner_idx]
            # Thus coeff of A_a in X_k[inner_idx] is U_4x4[k][block_idx]
            
            coeff_in_X = U_4x4[k][block_idx]
            if coeff_in_X == 0:
                continue
                
            for t in range(7):
                # U_2x2[k][t] is the vector for X in step t
                # We need the component corresponding to inner_idx
                coeff_in_U = U_2x2[k][t][inner_idx]
                if coeff_in_U == 0:
                    continue
                    
                col = k * 7 + t
                U[a][col] += coeff_in_X * coeff_in_U
                
    # Construct V
    for b in range(m * p):
        block_idx = b // 4
        inner_idx = b % 4
        
        for k in range(7):
            coeff_in_Y = V_4x4[k][block_idx]
            if coeff_in_Y == 0:
                continue
                
            for t in range(7):
                coeff_in_V = V_2x2[k][t][inner_idx]
                if coeff_in_V == 0:
                    continue
                    
                col = k * 7 + t
                V[b][col] += coeff_in_Y * coeff_in_V
                
    # Construct W
    for c in range(p * n):
        block_idx = c // 4
        inner_idx = c % 4
        
        for k in range(7):
            coeff_in_C = W_4x4[k][block_idx]
            if coeff_in_C == 0:
                continue
                
            for t in range(7):
                coeff_in_W = W_2x2[k][t][inner_idx]
                if coeff_in_W == 0:
                    continue
                    
                col = k * 7 + t
                W[c][col] += coeff_in_C * coeff_in_W
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
