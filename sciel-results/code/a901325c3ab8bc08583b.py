"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We split the 9 columns of B into two 4-column blocks and one 1-column block.
    # Block 1: 4x4x4 using Strassen (Rank 49)
    # Block 2: 4x4x4 using Strassen (Rank 49)
    # Block 3: 4x4x1 using Schoolbook (Rank 16)
    # Total Rank: 49 + 49 + 16 = 114
    
    def get_strassen_2x2():
        """Returns U, V, W for 2x2x2 Strassen decomposition."""
        # U: 4x7, V: 4x7, W: 4x7
        # Indices: A: 0,1,2,3 (row-major), B: 0,1,2,3, C: 0,1,2,3
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = 1
        
        # M3 = A00(B10-B11)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4 = A11(B01-B00)
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[1][3] = 1; W[3][3] = 1
        
        # M5 = (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[1][6] = -1
        
        return U, V, W

    def get_strassen_4x4():
        """Returns U, V, W for 4x4x4 Strassen decomposition (rank 49)."""
        U = [[0]*49 for _ in range(16)]
        V = [[0]*49 for _ in range(16)]
        W = [[0]*49 for _ in range(16)]
        
        u2, v2, w2 = get_strassen_2x2()
        
        # A11: indices 0,1,4,5
        # A12: indices 2,3,6,7
        # A21: indices 8,9,12,13
        # A22: indices 10,11,14,15
        # B11: indices 0,1,4,5
        # B12: indices 2,3,6,7
        # B21: indices 8,9,12,13
        # B22: indices 10,11,14,15
        # C11: indices 0,1,4,5
        # C12: indices 2,3,6,7
        # C21: indices 8,9,12,13
        # C22: indices 10,11,14,15
        
        # For each of the 7 Strassen products, we have a 2x2x2 decomposition.
        # Each 2x2x2 decomposition has 7 rank-1 terms.
        # Total: 7 * 7 = 49 rank-1 terms.
        
        # Let's define the linear combinations for each Mk.
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # For each Mk, we need to add the contributions to U, V, W.
        # The 2x2 Strassen U, V, W are defined for local indices 0,1,2,3.
        # We need to map these to the global indices for the specific blocks.
        
        def add_2x2_contribution(u_block, v_block, w_block, a_indices, b_indices, c_indices, rank_offset):
            """Add the 2x2 Strassen contribution to the global U, V, W."""
            for i in range(4):
                for k in range(7):
                    if u_block[i][k] != 0:
                        U[a_indices[i]][rank_offset + k] += u_block[i][k]
                    if v_block[i][k] != 0:
                        V[b_indices[i]][rank_offset + k] += v_block[i][k]
                    if w_block[i][k] != 0:
                        W[c_indices[i]][rank_offset + k] += w_block[i][k]
        
        # Define the index mappings for each block.
        # A11: [0, 1, 4, 5]
        # A12: [2, 3, 6, 7]
        # A21: [8, 9, 12, 13]
        # A22: [10, 11, 14, 15]
        # B11: [0, 1, 4, 5]
        # B12: [2, 3, 6, 7]
        # B21: [8, 9, 12, 13]
        # B22: [10, 11, 14, 15]
        # C11: [0, 1, 4, 5]
        # C12: [2, 3, 6, 7]
        # C21: [8, 9, 12, 13]
        # C22: [10, 11, 14, 15]
        
        # We need to handle the linear combinations carefully.
        # For M1 = (A11+A22)(B11+B22), the U part is A11+A22.
        # This means for each local index i in the 2x2 Strassen U, we add to both A11[i] and A22[i].
        # Similarly for V and W.
        
        # Let's define a helper to add a scaled 2x2 block.
        def add_scaled_2x2_block(u_block, v_block, w_block, a_indices, b_indices, c_indices, scale, rank_offset):
            for i in range(4):
                for k in range(7):
                    if u_block[i][k] != 0:
                        U[a_indices[i]][rank_offset + k] += scale * u_block[i][k]
                    if v_block[i][k] != 0:
                        V[b_indices[i]][rank_offset + k] += scale * v_block[i][k]
                    if w_block[i][k] != 0:
                        W[c_indices[i]][rank_offset + k] += scale * w_block[i][k]
        
        # M1: (A11+A22)(B11+B22)
        # U: A11 + A22, V: B11 + B22, W: C11 + C22
        # We can't just add two 2x2 blocks because the rank-1 terms are shared.
        # Actually, the Strassen algorithm computes M1 as a single 2x2 multiplication.
        # The inputs to this 2x2 multiplication are X1 = A11+A22 and Y1 = B11+B22.
        # So the U part for M1 is X1, which is A11+A22.
        # The V part is Y1, which is B11+B22.
        # The W part is the result Z1, which is added to C11 and C22.
        
        # Let's redefine the approach.
        # For each Mk, we define the linear combinations for Xk, Yk, and the target C blocks.
        
        # M1: X1 = A11 + A22, Y1 = B11 + B22, Z1 -> C11 + C22
        # M2: X2 = A21 + A22, Y2 = B11, Z2 -> C21 + C22 (with sign - for C22? No, Strassen: C21 += Z2, C22 -= Z2? Wait, let me check.)
        # Standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So:
        # M1: X1 = A11+A22, Y1 = B11+B22, C11 += M1, C22 += M1
        # M2: X2 = A21+A22, Y2 = B11, C21 += M2, C22 -= M2
        # M3: X3 = A11, Y3 = B21-B22, C12 += M3, C22 += M3
        # M4: X4 = A22, Y4 = B12-B11, C11 += M4, C21 += M4
        # M5: X5 = A11+A12, Y5 = B22, C11 -= M5, C12 += M5
        # M6: X6 = A21-A11, Y6 = B11+B12, C22 += M6
        # M7: X7 = A12-A22, Y7 = B21+B22, C11 += M7
        
        # Now, for each Mk, we use the 2x2 Strassen decomposition.
        # The 2x2 Strassen U, V, W are for a generic 2x2 multiplication.
        # We need to apply the linear combinations to the indices.
        
        # Let's define the index lists for each block.
        A11_idx = [0, 1, 4, 5]
        A12_idx = [2, 3, 6, 7]
        A21_idx = [8, 9, 12, 13]
        A22_idx = [10, 11, 14, 15]
        B11_idx = [0, 1, 4, 5]
        B12_idx = [2, 3, 6, 7]
        B21_idx = [8, 9, 12, 13]
        B22_idx = [10, 11, 14, 15]
        C11_idx = [0, 1, 4, 5]
        C12_idx = [2, 3, 6, 7]
        C21_idx = [8, 9, 12, 13]
        C22_idx = [10, 11, 14, 15]
        
        # Helper to add a linear combination of 2x2 blocks to U, V, W.
        def add_linear_combination_2x2(u_block, v_block, w_block, 
                                        u_indices, u_scales, 
                                        v_indices, v_scales, 
                                        w_indices, w_scales, 
                                        rank_offset):
            """
            u_block, v_block, w_block: 4x7 matrices from 2x2 Strassen.
            u_indices: list of index lists for A blocks.
            u_scales: list of scales for A blocks.
            v_indices: list of index lists for B blocks.
            v_scales: list of scales for B blocks.
            w_indices: list of index lists for C blocks.
            w_scales: list of scales for C blocks.
            rank_offset: offset in the rank dimension.
            """
            for i in range(4):
                for k in range(7):
                    # U part
                    for idx_list, scale in zip(u_indices, u_scales):
                        if u_block[i][k] != 0:
                            U[idx_list[i]][rank_offset + k] += scale * u_block[i][k]
                    # V part
                    for idx_list, scale in zip(v_indices, v_scales):
                        if v_block[i][k] != 0:
                            V[idx_list[i]][rank_offset + k] += scale * v_block[i][k]
                    # W part
                    for idx_list, scale in zip(w_indices, w_scales):
                        if w_block[i][k] != 0:
                            W[idx_list[i]][rank_offset + k] += scale * w_block[i][k]
        
        # M1: X1 = A11 + A22, Y1 = B11 + B22, C11 += M1, C22 += M1
        add_linear_combination_2x2(u2, v2, w2,
                                   [A11_idx, A22_idx], [1, 1],
                                   [B11_idx, B22_idx], [1, 1],
                                   [C11_idx, C22_idx], [1, 1],
                                   0)
        
        # M2: X2 = A21 + A22, Y2 = B11, C21 += M2, C22 -= M2
        add_linear_combination_2x2(u2, v2, w2,
                                   [A21_idx, A22_idx], [1, 1],
                                   [B11_idx], [1],
                                   [C21_idx, C22_idx], [1, -1],
                                   7)
        
        # M3: X3 = A11, Y3 = B21 - B22, C12 += M3, C22 += M3
        add_linear_combination_2x2(u2, v2, w2,
                                   [A11_idx], [1],
                                   [B21_idx, B22_idx], [1, -1],
                                   [C12_idx, C22_idx], [1, 1],
                                   14)
        
        # M4: X4 = A22, Y4 = B12 - B11, C11 += M4, C21 += M4
        add_linear_combination_2x2(u2, v2, w2,
                                   [A22_idx], [1],
                                   [B12_idx, B11_idx], [1, -1],
                                   [C11_idx, C21_idx], [1, 1],
                                   21)
        
        # M5: X5 = A11 + A12, Y5 = B22, C11 -= M5, C12 += M5
        add_linear_combination_2x2(u2, v2, w2,
                                   [A11_idx, A12_idx], [1, 1],
                                   [B22_idx], [1],
                                   [C11_idx, C12_idx], [-1, 1],
                                   28)
        
        # M6: X6 = A21 - A11, Y6 = B11 + B12, C22 += M6
        add_linear_combination_2x2(u2, v2, w2,
                                   [A21_idx, A11_idx], [1, -1],
                                   [B11_idx, B12_idx], [1, 1],
                                   [C22_idx], [1],
                                   35)
        
        # M7: X7 = A12 - A22, Y7 = B21 + B22, C11 += M7
        add_linear_combination_2x2(u2, v2, w2,
                                   [A12_idx, A22_idx], [1, -1],
                                   [B21_idx, B22_idx], [1, 1],
                                   [C11_idx], [1],
                                   42)
        
        return U, V, W

    # Assemble the full 4x4x9 decomposition.
    total_rank = 49 + 49 + 16
    U_full = [[0]*total_rank for _ in range(16)]
    V_full = [[0]*total_rank for _ in range(36)]
    W_full = [[0]*total_rank for _ in range(36)]
    
    U1, V1, W1 = get_strassen_4x4()
    U2, V2, W2 = get_strassen_4x4()
    
    # Block 1: 4x4x4 (B cols 0-3, C cols 0-3)
    for i in range(16):
        for k in range(49):
            U_full[i][k] = U1[i][k]
            V_full[i][k] = V1[i][k]
            W_full[i][k] = W1[i][k]
    
    # Block 2: 4x4x4 (B cols 4-7, C cols 4-7)
    for i in range(16):
        for k in range(49):
            U_full[i][49 + k] = U2[i][k]
            V_full[i + 4*4][49 + k] = V2[i][k]  # B rows 4-7
            W_full[i + 4*4][49 + k] = W2[i][k]  # C rows 4-7
    
    # Block 3: 4x4x1 (B col 8, C col 8)
    # Schoolbook
    rank_offset = 98
    for i in range(4):
        for j in range(4):
            col = rank_offset + i * 4 + j
            U_full[i * 4 + j][col] = 1
            V_full[j * 9 + 8][col] = 1
            W_full[8 * 4 + i][col] = 1
    
    return {"U": U_full, "V": V_full, "W": W_full}
# EVOLVE-BLOCK-END
