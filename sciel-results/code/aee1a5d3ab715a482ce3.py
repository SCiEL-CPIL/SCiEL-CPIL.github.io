import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    # We are using a block decomposition: 2x2 blocks of size 3x3.
    # Block indices: i, j in {0, 1}. Sub-block indices: x, y, z in {0, 1, 2}.
    # Global row index: 2*i + x? No, usually row-major.
    # Let's define the mapping.
    # A is 6x6. Block A_ij is 3x3.
    # A[2*i + x, 2*j + y] corresponds to block A_ij at position (x, y).
    
    # We will build U, V, W as lists of lists (or numpy arrays converted to lists) of shape (36, 189).
    # Rank = 7 (Strassen) * 27 (Schoolbook 3x3) = 189.
    
    rank = 189
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    # Define the 7 Strassen steps.
    # Each step k has:
    # U_k: 2x2 matrix (coefficients for A blocks)
    # V_k: 2x2 matrix (coefficients for B blocks)
    # W_k: 2x2 matrix (coefficients for C blocks)
    
    # M1 = (A00+A11)(B00+B11)
    U1 = [[1, 0], [0, 1]]
    V1 = [[1, 0], [0, 1]]
    W1 = [[1, 0], [0, 1]] # C00 += M1, C11 += M1
    
    # M2 = (A10+A11)B00
    U2 = [[0, 0], [1, 1]]
    V2 = [[1, 0], [0, 0]]
    W2 = [[0, 0], [1, 0]] # C10 += M2
    
    # M3 = A00(B01-B11)
    U3 = [[1, 0], [0, 0]]
    V3 = [[0, 1], [0, -1]]
    W3 = [[0, 1], [0, 1]] # C01 += M3, C11 += M3
    
    # M4 = A11(B10-B00)
    U4 = [[0, 0], [0, 1]]
    V4 = [[0, 0], [1, -1]]
    W4 = [[1, 0], [0, 1]] # C00 += M4, C10 += M4
    
    # M5 = (A00+A01)B11
    U5 = [[1, 1], [0, 0]]
    V5 = [[0, 0], [0, 1]]
    W5 = [[-1, 1], [0, 0]] # C00 -= M5, C01 += M5
    
    # M6 = (A10-A00)(B00+B01)
    U6 = [[1, 0], [-1, 0]]
    V6 = [[1, 1], [0, 0]]
    W6 = [[0, 0], [0, 1]] # C11 += M6
    
    # M7 = (A01-A11)(B10+B11)
    U7 = [[0, 1], [0, -1]]
    V7 = [[0, 0], [1, 1]]
    W7 = [[1, 0], [0, 0]] # C00 += M7
    
    steps = [
        (U1, V1, W1),
        (U2, V2, W2),
        (U3, V3, W3),
        (U4, V4, W4),
        (U5, V5, W5),
        (U6, V6, W6),
        (U7, V7, W7)
    ]
    
    # For each Strassen step, we perform a 3x3 matrix multiplication using schoolbook.
    # Schoolbook for 3x3:
    # C[x, z] = sum_y A[x, y] * B[y, z]
    # Rank 1 term for (x, y, z):
    # U_block: 3x3 matrix with 1 at (x, y), 0 elsewhere.
    # V_block: 3x3 matrix with 1 at (y, z), 0 elsewhere.
    # W_block: 3x3 matrix with 1 at (z, x), 0 elsewhere.
    
    col = 0
    for U_step, V_step, W_step in steps:
        for x in range(3):
            for y in range(3):
                for z in range(3):
                    # This is one rank-1 term in the 3x3 multiplication.
                    # It contributes to the global rank at index 'col'.
                    
                    # 1. Fill U (A part)
                    # U_step is 2x2. U_block is 3x3.
                    # The term is sum_{i,j} U_step[i,j] * U_block[i,j] ?
                    # No, the tensor is sum_r U_r \otimes V_r \otimes W_r.
                    # Here, for a specific Strassen step k and sub-term (x,y,z):
                    # The effective U matrix for the whole 6x6 is the Kronecker product of U_step (2x2) and U_block (3x3)?
                    # Actually, it's simpler:
                    # U_global[row, col] = sum over blocks...
                    # Let's just iterate over the non-zero entries of the combined matrices.
                    
                    # The "effective" U matrix for this rank-1 term is:
                    # U_eff[i, j] = U_step[i_block, j_block] * U_block[x, y]
                    # where i = 2*i_block + x, j = 2*j_block + y.
                    # Since U_block has only one non-zero entry at (x,y) with value 1,
                    # U_eff[i, j] is non-zero only if i_block, j_block match the non-zero entries of U_step.
                    # Specifically, for each non-zero (ib, jb) in U_step:
                    #   i = 2*ib + x
                    #   j = 2*jb + y
                    #   U[i, j] = U_step[ib, jb] * 1
                    
                    for ib in range(2):
                        for jb in range(2):
                            if U_step[ib][jb] != 0:
                                i = 2 * ib + x
                                j = 2 * jb + y
                                U[i * 6 + j][col] = U_step[ib][jb]
                                
                    # 2. Fill V (B part)
                    # V_eff[j, k] = V_step[jb, kb] * V_block[y, z]
                    # V_block has 1 at (y, z).
                    # So for each non-zero (jb, kb) in V_step:
                    #   j = 2*jb + y
                    #   k = 2*kb + z
                    #   V[j, k] = V_step[jb, kb] * 1
                    
                    for jb in range(2):
                        for kb in range(2):
                            if V_step[jb][kb] != 0:
                                j = 2 * jb + y
                                k = 2 * kb + z
                                V[j * 6 + k][col] = V_step[jb][kb]
                                
                    # 3. Fill W (C part)
                    # W_eff[k, i] = W_step[kb, ib] * W_block[z, x]
                    # W_block has 1 at (z, x).
                    # So for each non-zero (kb, ib) in W_step:
                    #   k = 2*kb + z
                    #   i = 2*ib + x
                    #   W[k, i] = W_step[kb, ib] * 1
                    # Note: The interface requires W[k*n+i, r]. Here n=6.
                    # So index is k*6 + i.
                    
                    for kb in range(2):
                        for ib in range(2):
                            if W_step[kb][ib] != 0:
                                k = 2 * kb + z
                                i = 2 * ib + x
                                W[k * 6 + i][col] = W_step[kb][ib]
                    
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
