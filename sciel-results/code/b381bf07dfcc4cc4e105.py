"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 22 Decomposition
    # Split m=3 into m1=2 and m2=1.
    # Part 1: 2x2x4 multiplication (Rank 14, composed of two 2x2x2 Strassen blocks)
    # Part 2: 2x1x4 multiplication (Rank 8)
    
    rank = 22
    U = [[0] * rank for _ in range(n * m)] # 6 x 22
    V = [[0] * rank for _ in range(m * p)] # 12 x 22
    W = [[0] * rank for _ in range(p * n)] # 8 x 22
    
    # Helper to add Strassen 2x2x2 block
    # Maps local indices (i, j, k) to global indices
    # A is n x m1, B is m1 x p, C is p x n
    # Here we use A[:, 0:2], B[0:2, :], C[:, :]
    # This is effectively two 2x2x2 multiplications side-by-side in B's columns?
    # No, B is 2x4. We split B into B_left (2x2) and B_right (2x2).
    # C = A B_left + A B_right.
    
    # Strassen Factors for 2x2x2
    # U (4x7), V (4x7), W (4x7)
    # Indices for A: 0,1,2,3 (row-major 2x2)
    # Indices for B: 0,1,2,3 (row-major 2x2)
    # Indices for C: 0,1,2,3 (row-major 2x2)
    
    # We will place two Strassen blocks.
    # Block 1: B cols 0,1. C cols 0,1.
    # Block 2: B cols 2,3. C cols 2,3.
    
    def add_strassen_block(offset_u, offset_v, offset_w, col_offset_b, col_offset_c):
        # offset_u: row offset in U (A)
        # offset_v: row offset in V (B)
        # offset_w: row offset in W (C)
        # col_offset_b: column offset in B (for V index mapping)
        # col_offset_c: column offset in C (for W index mapping)
        
        # Strassen 7 terms
        # Term 0: (A00+A11)(B00+B11)
        # Term 1: (A10+A11)B00
        # Term 2: A00(B01-B11)
        # Term 3: A11(B10-B00)
        # Term 4: (A00+A01)B11
        # Term 5: (A10-A00)(B00+B01)
        # Term 6: (A01-A11)(B10+B11)
        
        # U vectors (A)
        # A00 is idx 0, A01 is idx 1, A10 is idx 2, A11 is idx 3
        U_vecs = [
            [1, 1, 0, 0], # A00+A11
            [0, 0, 1, 1], # A10+A11
            [1, 0, 0, 0], # A00
            [0, 0, 0, 1], # A11
            [1, 1, 0, 0], # A00+A01? No. A00+A01 is [1,1,0,0]? No.
                         # A is [[A00, A01], [A10, A11]].
                         # A00+A01 is not a standard vector in Strassen.
                         # Strassen uses:
                         # M5 = (A00+A01)B11. Vector: [1, 1, 0, 0]? No.
                         # A00 is (0,0), A01 is (0,1).
                         # A00+A01 = [1, 1, 0, 0] in row-major?
                         # Yes, if A00 is index 0, A01 is index 1.
                         # So [1, 1, 0, 0].
            [1, 0, -1, 0], # A10-A00. A10 is idx 2, A00 is idx 0. [ -1, 0, 1, 0]?
                          # Wait, A10 is (1,0). Index 2. A00 is (0,0). Index 0.
                          # A10 - A00 = [ -1, 0, 1, 0 ].
            [0, 1, 0, -1]  # A01-A11. A01 idx 1, A11 idx 3. [0, 1, 0, -1].
        ]
        
        # V vectors (B)
        # B00 idx 0, B01 idx 1, B10 idx 2, B11 idx 3
        V_vecs = [
            [1, 1, 0, 0], # B00+B11? No. B00+B11 is [1,0,0,1].
                         # M1 = (A00+A11)(B00+B11).
                         # B00+B11 = [1, 0, 0, 1].
            [1, 0, 0, 0], # B00
            [0, 1, 0, -1], # B01-B11. B01 idx 1, B11 idx 3. [0,1,0,-1].
            [0, -1, 1, 0], # B10-B00. B10 idx 2, B00 idx 0. [ -1, 0, 1, 0 ].
            [0, 0, 0, 1], # B11
            [1, 1, 0, 0], # B00+B01. [1,1,0,0].
            [0, 0, 1, 1]  # B10+B11. [0,0,1,1].
        ]
        # Correcting V_vecs[0]: B00+B11 is [1,0,0,1].
        V_vecs[0] = [1, 0, 0, 1]
        
        # W vectors (C)
        # C00 idx 0, C01 idx 1, C10 idx 2, C11 idx 3
        # C = M1 - M4 - M6, C01 = M3 + M5, C10 = M2 + M4, C11 = M1 - M3 - M5 + M7
        # Wait, standard Strassen output:
        # C00 = M1 - M4 - M6
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 - M5 + M7
        
        # We need W vectors for each M term.
        # W for M1: contributes to C00, C11.
        # W for M2: contributes to C10.
        # W for M3: contributes to C01, C11.
        # W for M4: contributes to C00, C10.
        # W for M5: contributes to C01, C11.
        # W for M6: contributes to C00.
        # W for M7: contributes to C11.
        
        W_vecs = [
            [1, 0, 0, 1], # M1: C00+1, C11+1
            [0, 0, 1, 0], # M2: C10+1
            [0, 1, 0, -1], # M3: C01+1, C11-1
            [-1, 0, 1, 0], # M4: C00-1, C10+1
            [0, 1, 0, -1], # M5: C01+1, C11-1
            [-1, 0, 0, 0], # M6: C00-1
            [0, 0, 0, 1]   # M7: C11+1
        ]
        
        for t in range(7):
            col = offset_u + t # No, column index in U,V,W matrices
            # The column index is the rank index.
            # We are filling U, V, W matrices.
            # U has shape (4, 7) locally.
            # We map local row r to global row offset_u + r.
            # We map local col c to global col offset_u + c?
            # No, the rank index is the same for U, V, W.
            # So for term t, the rank index is offset_rank + t.
            
            rank_idx = offset_rank + t
            
            # Fill U
            for r in range(4):
                U[offset_u + r][rank_idx] = U_vecs[t][r]
                
            # Fill V
            # B local indices 0,1,2,3 map to global B indices.
            # B is 2x4.
            # B_left is cols 0,1. B_right is cols 2,3.
            # If this is B_left, col_offset_b = 0.
            # Local B row-major: 0(0,0), 1(0,1), 2(1,0), 3(1,1).
            # Global B row-major:
            # Row 0: 0(0,0), 1(0,1), 2(0,2), 3(0,3)
            # Row 1: 4(1,0), 5(1,1), 6(1,2), 7(1,3)
            # So local 0 -> global 0. Local 1 -> global 1.
            # Local 2 -> global 4. Local 3 -> global 5.
            
            for r in range(4):
                # Map local B index r to global B index
                # local r: 0,1,2,3
                # if r < 2: global = r
                # else: global = r + 2 (since row 1 starts at 4, local 2 is 1,0 -> global 4)
                # Wait.
                # Local B is 2x2.
                # Global B is 2x4.
                # If we are using B[0:2, 0:2], then:
                # Local (0,0) -> Global (0,0) idx 0.
                # Local (0,1) -> Global (0,1) idx 1.
                # Local (1,0) -> Global (1,0) idx 4.
                # Local (1,1) -> Global (1,1) idx 5.
                
                if r < 2:
                    g_idx = r + col_offset_b # col_offset_b is 0 for left, 2 for right?
                    # If col_offset_b is 2, global col is 2,3.
                    # Local (0,0) -> Global (0,2) idx 2.
                    # Local (0,1) -> Global (0,3) idx 3.
                    # Local (1,0) -> Global (1,2) idx 6.
                    # Local (1,1) -> Global (1,3) idx 7.
                    # So if r < 2: g_idx = r + col_offset_b.
                    # If r >= 2: g_idx = r + 2 + col_offset_b.
                else:
                    g_idx = r + 2 + col_offset_b
                    
                V[offset_v + g_idx][rank_idx] = V_vecs[t][r]

            # Fill W
            # C is 4x2.
            # C_left is cols 0,1. C_right is cols 2,3.
            # Local C is 2x2.
            # Global C row-major:
            # Row 0: 0(0,0), 1(0,1), 2(0,2), 3(0,3)
            # Row 1: 4(1,0), 5(1,1), 6(1,2), 7(1,3)
            
            for r in range(4):
                if r < 2:
                    g_idx = r + col_offset_c
                else:
                    g_idx = r + 2 + col_offset_c
                    
                W[offset_w + g_idx][rank_idx] = W_vecs[t][r]

    # Block 1: A[:, 0:2], B[0:2, 0:2], C[:, 0:2]
    # U offset 0, V offset 0, W offset 0
    # B col offset 0, C col offset 0
    # Rank offset 0
    add_strassen_block(0, 0, 0, 0, 0, 0)
    
    # Block 2: A[:, 0:2], B[0:2, 2:4], C[:, 2:4]
    # U offset 0, V offset 0, W offset 0
    # B col offset 2, C col offset 2
    # Rank offset 7
    add_strassen_block(0, 0, 0, 2, 2, 7)
    
    # Part 2: 2x1x4 multiplication
    # A[:, 2], B[2, :], C[:, :]
    # Rank offset 14
    # U rows 4,5 (A col 2)
    # V rows 8,9,10,11 (B row 2)
    # W rows 0..7
    
    # Terms: A_i2 * B_2k for all i,k
    # i in 0,1. k in 0,1,2,3.
    # 8 terms.
    
    for i in range(2):
        for k in range(4):
            rank_idx = 14 + i * 4 + k
            
            # U: A_i2. Global A index: i * 3 + 2.
            U[i * 3 + 2][rank_idx] = 1
            
            # V: B_2k. Global B index: 2 * 4 + k = 8 + k.
            V[8 + k][rank_idx] = 1
            
            # W: C_ki. Global C index: k * 2 + i.
            W[k * 2 + i][rank_idx] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
