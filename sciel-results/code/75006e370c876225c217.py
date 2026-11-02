"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    # Rank 65 decomposition:
    # 49 terms for 4x4x4 part
    # 16 terms for 4x4x1 part
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65
    
    # --- Part 1: 4x4x4 Decomposition (Rank 49) ---
    # Strassen terms for 2x2
    # U_vecs, V_vecs, W_vecs for scalar 2x2 multiplication
    # Indices 0..3 for 2x2 elements
    U2 = [
        [1, 0, 0, 1], # M1: A00+A11
        [0, 0, 1, 1], # M2: A10+A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00+A01
        [-1, 0, 1, 0],# M6: A10-A00
        [0, 1, 0, -1] # M7: A01-A11
    ]
    V2 = [
        [1, 0, 0, 1], # M1: B00+B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01-B11
        [-1, 0, 1, 0],# M4: B10-B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00+B01
        [0, 0, 1, 1]  # M7: B10+B11
    ]
    W2 = [
        [1, 0, 0, 1], # M1: C00+C11
        [0, 0, 1, -1],# M2: C10-C11
        [-1, 1, 0, 0],# M3: -C00+C01
        [-1, 0, 1, 0],# M4: -C00+C10
        [0, 1, 0, 1], # M5: C01+C11
        [0, 0, 0, -1],# M6: -C11
        [1, 0, 0, -1] # M7: C00-C11
    ]
    
    # Block coefficients for 4x4 (2x2 blocks of 2x2 matrices)
    # Block indices: 0:00, 1:01, 2:10, 3:11
    # U_blocks[k][b] = coeff of block b in k-th Strassen term
    U_blocks = [
        [1, 0, 0, 1], # S1: A00+A11
        [0, 0, 1, 1], # S2: A10+A11
        [1, 0, 0, 0], # S3: A00
        [0, 0, 0, 1], # S4: A11
        [1, 1, 0, 0], # S5: A00+A01
        [-1, 0, 1, 0],# S6: A10-A00
        [0, 1, 0, -1] # S7: A01-A11
    ]
    V_blocks = [
        [1, 0, 0, 1], # S1: B00+B11
        [1, 0, 0, 0], # S2: B00
        [0, 1, 0, -1],# S3: B01-B11
        [-1, 0, 1, 0],# S4: B10-B00
        [0, 0, 0, 1], # S5: B11
        [1, 1, 0, 0], # S6: B00+B01
        [0, 0, 1, 1]  # S7: B10+B11
    ]
    # W_blocks[k][b] = coeff of block b in reconstruction of k-th Strassen term
    # C00 = S1 - S3 - S4 + S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S2 + S5 - S6
    W_blocks = [
        [1, 0, 0, 1], # S1: C00+C11
        [0, 0, 1, -1],# S2: C10-C11
        [-1, 1, 0, 0],# S3: -C00+C01
        [-1, 0, 1, 0],# S4: -C00+C10
        [0, 1, 0, 1], # S5: C01+C11
        [0, 0, 0, -1],# S6: -C11
        [1, 0, 0, -1] # S7: C00-C11
    ]
    
    for k in range(7): # Block term
        for l in range(7): # Scalar term
            col = k * 7 + l
            
            # U part
            for b in range(4):
                coeff_block = U_blocks[k][b]
                if coeff_block != 0:
                    # Elements of block b in 4x4 matrix
                    # Block 0: indices 0,1,2,3
                    # Block 1: indices 4,5,6,7
                    # Block 2: indices 8,9,10,11
                    # Block 3: indices 12,13,14,15
                    base_idx = b * 4
                    for j in range(4):
                        coeff_scalar = U2[l][j]
                        if coeff_scalar != 0:
                            U[base_idx + j][col] += coeff_block * coeff_scalar
            
            # V part
            for b in range(4):
                coeff_block = V_blocks[k][b]
                if coeff_block != 0:
                    base_idx = b * 4
                    for j in range(4):
                        coeff_scalar = V2[l][j]
                        if coeff_scalar != 0:
                            V[base_idx + j][col] += coeff_block * coeff_scalar
            
            # W part
            for b in range(4):
                coeff_block = W_blocks[k][b]
                if coeff_block != 0:
                    base_idx = b * 4
                    for j in range(4):
                        coeff_scalar = W2[l][j]
                        if coeff_scalar != 0:
                            W[base_idx + j][col] += coeff_block * coeff_scalar

    # --- Part 2: 4x4x1 Decomposition (Rank 16) ---
    # A (4x4) * B_right (4x1) -> C_right (4x1)
    # C_right[i] = sum_j A[i,j] B_right[j]
    # 16 terms, one for each (i,j) pair
    for i in range(4):
        for j in range(4):
            col = 49 + (i * 4 + j)
            
            # U: A[i,j]
            U[i * 4 + j][col] = 1
            
            # V: B[j, 4] (5th column, index 4 in B's 4x5 matrix)
            # B index is j * 5 + 4
            V[j * 5 + 4][col] = 1
            
            # W: C[i, 4] (5th column, index 4 in C's 5x4 matrix)
            # C index is 4 * 4 + i
            W[4 * 4 + i][col] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
