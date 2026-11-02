import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Decomposition:
    # Split m=5 into 4+1.
    # Split p=6 into 4+2.
    # 
    # Term 1: 2 x 4 x 4 (Rank 28)
    # Term 2: 2 x 4 x 2 (Rank 14)
    # Term 3: 2 x 1 x 6 (Rank 12)
    # Total Rank: 54
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)] # 10 x 54
    V = [[0] * rank for _ in range(m * p)] # 30 x 54
    W = [[0] * rank for _ in range(p * n)] # 12 x 54
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen decomposition
    # U indices: i*m + j (i in 0..1, j in 0..1)
    # V indices: j*p + k (j in 0..1, k in 0..1) -- Note: p is local dim
    # W indices: k*n + i (k in 0..1, i in 0..1)
    
    # Strassen factors for 2x2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    strassen_U = [
        [1, 0, 0, 0, 1, 0, 0], # A11
        [0, 1, 1, 0, 0, -1, 0], # A21 (M2, M6) -- Wait, standard Strassen
        [0, 0, 1, 1, 0, 0, 1], # A22 (M1, M4, M7)
        [0, 0, 0, 0, 1, 1, 0]  # A12 (M5, M7) -- Indices need care
    ]
    # Let's use the standard 7 factors explicitly for U, V, W.
    # U corresponds to A, V to B, W to C.
    # Factor r: u_r (2x2), v_r (2x2), w_r (2x2)
    
    # r=1: M1
    # u1: A11+A22 -> [1, 0; 0, 1]
    # v1: B11+B22 -> [1, 0; 0, 1]
    # w1: C11+1, C22+1 -> [1, 0; 0, 1]
    
    # r=2: M2
    # u2: A21+A22 -> [0, 0; 1, 1]
    # v2: B11 -> [1, 0; 0, 0]
    # w2: C21+1 -> [0, 0; 1, 0]
    
    # r=3: M3
    # u3: A11 -> [1, 0; 0, 0]
    # v3: B21-B22 -> [0, 0; 1, -1]
    # w3: C12+1, C22+1 -> [0, 1; 0, 1]
    
    # r=4: M4
    # u4: A22 -> [0, 0; 0, 1]
    # v4: B12-B11 -> [-1, 1; 0, 0]
    # w4: C11+1, C21+1 -> [1, 0; 1, 0]
    
    # r=5: M5
    # u5: A11+A12 -> [1, 1; 0, 0]
    # v5: B22 -> [0, 0; 0, 1]
    # w5: C11-1, C12+1 -> [-1, 1; 0, 0]
    
    # r=6: M6
    # u6: A21-A11 -> [-1, 0; 1, 0]
    # v6: B11+B12 -> [1, 1; 0, 0]
    # w6: C22+1 -> [0, 0; 0, 1]
    
    # r=7: M7
    # u7: A12-A22 -> [0, 1; -1, 0]
    # v7: B21+B22 -> [0, 0; 1, 1]
    # w7: C11+1, C22+1 -> [1, 0; 0, 1]
    
    factors = [
        # u, v, w (2x2 matrices)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # 1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # 2
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]), # 3
        ([0, 0, 0, 1], [-1, 1, 0, 0], [1, 0, 1, 0]), # 4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # 5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # 6
        ([0, 1, -1, 0], [0, 0, 1, 1], [1, 0, 0, 1])  # 7
    ]
    
    def add_strassen(offset_u, offset_v, offset_w):
        nonlocal col
        for r in range(7):
            u, v, w = factors[r]
            for i in range(2):
                for j in range(2):
                    # U index: (offset_u + i) * m + (offset_v_j + j) ? 
                    # No, U is indexed by [row, col] of A.
                    # A is 2x2.
                    # Global U index: row_in_A * m + col_in_A
                    # Here A is a 2x2 block.
                    # row_in_A = offset_u + i
                    # col_in_A = offset_v_j + j ? No, col_in_A is the middle index.
                    
                    # Let's map carefully.
                    # Term 1: 2x4x4.
                    # A is 2x4. Rows 0,1. Cols 0..3.
                    # B is 4x4. Rows 0..3. Cols 0..3.
                    # C is 2x4. Rows 0,1. Cols 0..3.
                    
                    # We split A into A1 (2x2) and A2 (2x2).
                    # A1 cols 0,1. A2 cols 2,3.
                    # B split into B11 (2x2), B12 (2x2), B21 (2x2), B22 (2x2).
                    # B11 rows 0,1 cols 0,1.
                    # B12 rows 0,1 cols 2,3.
                    # B21 rows 2,3 cols 0,1.
                    # B22 rows 2,3 cols 2,3.
                    
                    # C11 (2x2) rows 0,1 cols 0,1.
                    # C12 (2x2) rows 0,1 cols 2,3.
                    # C21 (2x2) rows 2,3 cols 0,1.
                    # C22 (2x2) rows 2,3 cols 2,3.
                    
                    # Strassen combines A1, A2 with B11, B12, B21, B22 to produce C11, C12, C21, C22.
                    
                    # For each factor r:
                    # U (A part): 2x2 matrix.
                    # V (B part): 2x2 matrix.
                    # W (C part): 2x2 matrix.
                    
                    # We need to place these into the global U, V, W.
                    
                    # U global index: (row_A, col_A)
                    # V global index: (row_B, col_B)
                    # W global index: (col_C, row_C) -- Trace convention W[k*n+i, r]
                    
                    # For A1: row_A = 0,1. col_A = 0,1.
                    # For A2: row_A = 0,1. col_A = 2,3.
                    # For B11: row_B = 0,1. col_B = 0,1.
                    # For B12: row_B = 0,1. col_B = 2,3.
                    # For B21: row_B = 2,3. col_B = 0,1.
                    # For B22: row_B = 2,3. col_B = 2,3.
                    # For C11: col_C = 0,1. row_C = 0,1.
                    # For C12: col_C = 2,3. row_C = 0,1.
                    # For C21: col_C = 0,1. row_C = 2,3.
                    # For C22: col_C = 2,3. row_C = 2,3.
                    
                    # The Strassen factors u, v, w are 2x2.
                    # u[i,j] adds to U[offset_u+i, offset_u_col+j]
                    # v[i,j] adds to V[offset_v_row+i, offset_v_col+j]
                    # w[i,j] adds to W[(offset_w_col+j)*n + (offset_w_row+i)]
                    
                    # We have 4 Strassen instances for 2x4x4?
                    # No, 2x4x4 is ONE instance of 2x2x2 Strassen?
                    # No, 2x4x4 requires 4 multiplications of 2x2x2 if we don't share.
                    # But Strassen shares!
                    # Strassen for 2x2x2 takes 7 ops.
                    # For 2x4x4, we can view it as 2x(2+2)x(2+2).
                    # This is NOT a single 2x2x2 Strassen.
                    # It is 4 blocks of 2x2x2.
                    # Can we apply Strassen to the 2x4x4 as a whole?
                    # No, Strassen is for 2x2x2.
                    # So we must use 4 separate Strassen decompositions?
                    # If we use 4 separate Strassen decompositions, rank is 4*7=28.
                    
                    # Let's implement 4 separate Strassen decompositions for 2x4x4.
                    # 1. A1*B11 -> C11_part1
                    # 2. A2*B21 -> C11_part2
                    # 3. A1*B12 -> C12_part1
                    # 4. A2*B22 -> C12_part2
                    # Wait, C11 = A1*B11 + A2*B21.
                    # So we need to add the results.
                    # This means we can just run Strassen for each pair?
                    # No, Strassen computes C = AB for 2x2.
                    # Here we have sums.
                    # We can just treat A1*B11 as a 2x2x2 multiplication.
                    # And A2*B21 as another 2x2x2 multiplication.
                    # Their sum is C11.
                    # So we just add the U, V, W factors for each.
                    
                    # So for 2x4x4:
                    # Pair 1: A1 (cols 0,1) * B11 (rows 0,1, cols 0,1) -> C11 (rows 0,1, cols 0,1)
                    # Pair 2: A2 (cols 2,3) * B21 (rows 2,3, cols 0,1) -> C11 (rows 0,1, cols 0,1) -- Wait, B21 is rows 2,3. A2 is cols 2,3.
                    # A2 (2x2) * B21 (2x2) -> 2x2.
                    # This adds to C11.
                    
                    # So we run Strassen for (A1, B11, C11) and (A2, B21, C11).
                    # And (A1, B12, C12) and (A2, B22, C12).
                    
                    # Total 4 Strassen decompositions.
                    
                    # For 2x4x2:
                    # A1 (cols 0,1) * B1 (rows 0,1, cols 0,1) -> C1 (rows 0,1, cols 0,1)
                    # A2 (cols 2,3) * B2 (rows 2,3, cols 0,1) -> C2 (rows 0,1, cols 0,1)
                    # Wait, C is 2x2.
                    # C = A1*B1 + A2*B2.
                    # So 2 Strassen decompositions.
                    
                    # Total 6 Strassen decompositions.
                    # Rank 6 * 7 = 42.
                    # Plus 2x1x6 (Rank 12).
                    # Total 54.
                    
                    # Implementation:
                    # Define offsets for each of the 6 Strassen instances.
                    
                    instances = [
                        # (u_row_off, u_col_off, v_row_off, v_col_off, w_row_off, w_col_off)
                        # u_col_off is the offset in the middle dimension for A
                        # v_row_off is the offset in the middle dimension for B
                        # w_col_off is the offset in the output columns for C
                        # w_row_off is the offset in the output rows for C
                        
                        # 1. A1 * B11 -> C11
                        (0, 0, 0, 0, 0, 0),
                        # 2. A2 * B21 -> C11
                        (0, 2, 2, 0, 0, 0),
                        # 3. A1 * B12 -> C12
                        (0, 0, 0, 2, 0, 2),
                        # 4. A2 * B22 -> C12
                        (0, 2, 2, 2, 0, 2),
                        # 5. A1 * B1 -> C1 (for 2x4x2 part)
                        (0, 0, 0, 4, 0, 4), # B rows 0,1, cols 4,5? No, 2x4x2 uses cols 4,5 of B?
                        # Wait, p=6. Split 4+2.
                        # B is 4x6.
                        # B1 is 4x2 (cols 4,5).
                        # B1 rows 0,1. B2 rows 2,3.
                        # So B1 is rows 0,1, cols 4,5.
                        # B2 is rows 2,3, cols 4,5.
                        (0, 0, 0, 4, 0, 4),
                        # 6. A2 * B2 -> C2
                        (0, 2, 2, 4, 0, 4)
                    ]
                    
                    for inst in instances:
                        u_ro, u_co, v_ro, v_co, w_ro, w_co = inst
                        for r in range(7):
                            u, v, w = factors[r]
                            for i in range(2):
                                for j in range(2):
                                    U[(u_ro + i) * m + (u_co + j)][col] = u[i*2+j]
                                    V[(v_ro + i) * p + (v_co + j)][col] = v[i*2+j]
                                    W[(w_co + j) * n + (w_ro + i)][col] = w[i*2+j]
                            col += 1
                    
    add_strassen(0, 0, 0, 0, 0, 0) # This function call is wrong, I defined instances inside.
    
    # Let's refactor to just loop over instances directly.
    col = 0
    for inst in instances:
        u_ro, u_co, v_ro, v_co, w_ro, w_co = inst
        for r in range(7):
            u, v, w = factors[r]
            for i in range(2):
                for j in range(2):
                    U[(u_ro + i) * m + (u_co + j)][col] = u[i*2+j]
                    V[(v_ro + i) * p + (v_co + j)][col] = v[i*2+j]
                    W[(w_co + j) * n + (w_ro + i)][col] = w[i*2+j]
            col += 1
    
    # Add 2x1x6 part
    # A is 2x1 (col 4). B is 1x6 (row 4).
    # C is 2x6.
    # C_ij = A_i B_j.
    # Rank 12.
    for i in range(2):
        for j in range(6):
            U[i * m + 4][col] = 1
            V[4 * p + j][col] = 1
            W[j * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
