"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition using recursive Strassen
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 2x2 Strassen decomposition coefficients
    # U2[i, r], V2[j, r], W2[k, r]
    # i, j, k are indices in 0..3 for 2x2 matrices
    # r is 0..6
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # r=0: (A00+A11)(B00+B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # r=1: (A10+A11)B00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1 # C10
    
    # r=2: A00(B01-B11)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1 # C01, C11
    
    # r=3: A11(B10-B00)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1 # C00, C10
    
    # r=4: (A00+A01)B11
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1 # C00, C01
    
    # r=5: (A10-A00)(B00+B01)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1 # C11
    
    # r=6: (A01-A11)(B10+B11)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1 # C00
    
    # Outer Strassen products for 4x4
    # Blocks: 00, 01, 10, 11
    # A00: rows 0,1 cols 0,1 -> indices 0,1,4,5
    # A01: rows 0,1 cols 2,3 -> indices 2,3,6,7
    # A10: rows 2,3 cols 0,1 -> indices 8,9,12,13
    # A11: rows 2,3 cols 2,3 -> indices 10,11,14,15
    
    # Mapping block to local 2x2 indices
    # Block 00: local (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # Global indices:
    # A00: 0,1,4,5
    # A01: 2,3,6,7
    # A10: 8,9,12,13
    # A11: 10,11,14,15
    
    block_indices = {
        0: [0, 1, 4, 5],   # A00
        1: [2, 3, 6, 7],   # A01
        2: [8, 9, 12, 13], # A10
        3: [10, 11, 14, 15] # A11
    }
    
    # Outer Strassen coefficients
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    outer_A = [
        {0: 1, 3: 1},      # M1
        {2: 1, 3: 1},      # M2
        {0: 1},            # M3
        {3: 1},            # M4
        {0: 1, 1: 1},      # M5
        {2: 1, 0: -1},     # M6
        {1: 1, 3: -1}      # M7
    ]
    
    outer_B = [
        {0: 1, 3: 1},      # M1
        {0: 1},            # M2
        {1: 1, 3: -1},     # M3
        {2: 1, 0: -1},     # M4
        {3: 1},            # M5
        {0: 1, 1: 1},      # M6
        {2: 1, 3: 1}       # M7
    ]
    
    # Outer W coefficients
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # W_outer[block][m]
    # Block 0: C00, Block 1: C01, Block 2: C10, Block 3: C11
    outer_W = [
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0]    # C11
    ]
    
    col = 0
    for k in range(7):
        for r in range(7):
            # Combine coefficients
            # U: for each global index i in A
            # U[i][col] = sum_{b, local_i} outer_A[k][b] * U2[local_i][r]
            
            # We need to map local indices to global indices
            # local 0,1,2,3 correspond to the 4 entries of the block
            
            for b in range(4):
                if b not in outer_A[k]:
                    continue
                coeff_A = outer_A[k][b]
                if coeff_A == 0:
                    continue
                # Map local indices to global
                for local_i in range(4):
                    if U2[local_i][r] != 0:
                        global_i = block_indices[b][local_i]
                        U[global_i][col] += coeff_A * U2[local_i][r]
                        
            for c in range(4):
                if c not in outer_B[k]:
                    continue
                coeff_B = outer_B[k][c]
                if coeff_B == 0:
                    continue
                for local_j in range(4):
                    if V2[local_j][r] != 0:
                        global_j = block_indices[c][local_j]
                        V[global_j][col] += coeff_B * V2[local_j][r]
                        
            # W: The product P_kr contributes to C_k
            # C_k is a block.
            # The local entry (local_k, local_i) of C_k contributes to global C.
            # W2[local_k, local_i][r] gives the coefficient.
            # This local entry belongs to block k.
            # The block k is added to the final C with coefficient outer_W[block][k].
            
            for local_k in range(4):
                if W2[local_k][r] != 0:
                    # local_k is the index in the 2x2 block C_k
                    # We need to map local_k to the global index in C
                    # The block index for C_k is determined by the outer product k?
                    # No, C_k is one of the 4 blocks of C.
                    # Wait, the outer Strassen algorithm computes 4 blocks of C.
                    # C00 = M1 + M4 - M5 + M7
                    # So M1 contributes to C00.
                    # M2 contributes to C10.
                    # M3 contributes to C01 and C11.
                    # M4 contributes to C00 and C10.
                    # M5 contributes to C00 and C01.
                    # M6 contributes to C11.
                    # M7 contributes to C00.
                    
                    # So for a given outer product k, it contributes to specific blocks.
                    # We need to know which block M_k belongs to.
                    # Actually, M_k is a 2x2 matrix.
                    # The entries of M_k are added to the corresponding entries of the target blocks.
                    
                    # Let's use outer_W to determine the contribution.
                    # outer_W[block][k] is the coefficient of M_k in C_block.
                    
                    for block in range(4):
                        coeff_W_outer = outer_W[block][k]
                        if coeff_W_outer == 0:
                            continue
                        
                        # local_k is the index in the 2x2 matrix M_k
                        # We need to map local_k to the global index in C_block
                        # local_k 0,1,2,3 -> global indices in block
                        global_k = block_indices[block][local_k]
                        
                        W[global_k][col] += coeff_W_outer * W2[local_k][r]
                        
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
