import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition in parts.
    # Part 1: A (4x4) * B1 (4x4) -> C1 (4x4). Rank 49.
    # Part 2: A (4x4) * B2 (4x4) -> C2 (4x4). Rank 49.
    # Part 3: A (4x4) * B3 (4x1) -> C3 (4x1). Rank 16.
    
    total_rank = 49 + 49 + 16
    U = np.zeros((n*m, total_rank), dtype=float)
    V = np.zeros((m*p, total_rank), dtype=float)
    W = np.zeros((p*n, total_rank), dtype=float)
    
    # Helper to add a 2x2x2 Strassen decomposition
    # a_idx: indices for A blocks (2x2) -> 4 indices
    # b_idx: indices for B blocks (2x2) -> 4 indices
    # c_idx: indices for C blocks (2x2) -> 4 indices
    # base_col: starting column in U, V, W
    def add_strassen_2x2(a_row_base, b_row_base, c_row_base, base_col):
        # Strassen 2x2x2 has rank 7
        # U, V, W for 2x2x2
        # U shape (4, 7), V shape (4, 7), W shape (4, 7)
        # Indices 0..3 for 2x2 matrix: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Coefficients for U (A part)
        # Rows: A00, A01, A10, A11
        u_2x2 = [
            [1, 0, 1, 0, 1, -1, 0], # A00
            [0, 0, 0, 0, 1, 0, 1],  # A01
            [0, 1, 0, 0, 0, 1, 0],  # A10
            [1, 1, 0, 1, 0, 0, -1]  # A11
        ]
        
        # Coefficients for V (B part)
        # Rows: B00, B01, B10, B11
        v_2x2 = [
            [1, 1, 0, -1, 0, 1, 0], # B00
            [0, 0, 1, 0, 0, 1, 0],  # B01
            [0, 0, 0, 1, 0, 0, 1],  # B10
            [1, 0, -1, 0, 1, 0, 1]  # B11
        ]
        
        # Coefficients for W (C part)
        # Rows: C00, C01, C10, C11
        w_2x2 = [
            [1, 0, 0, 1, -1, 0, 1], # C00 = M1 + M4 - M5 + M7
            [0, 0, 1, 0, 1, 0, 0],  # C01 = M3 + M5
            [0, 1, 0, 1, 0, 0, 0],  # C10 = M2 + M4
            [1, -1, 1, 0, 0, 1, 0]  # C11 = M1 - M2 + M3 + M6
        ]
        
        # Map local indices to global indices
        # A local 0,1,2,3 -> global a_row_base + 0, 1, 2, 3
        # B local 0,1,2,3 -> global b_row_base + 0, 1, 2, 3
        # C local 0,1,2,3 -> global c_row_base + 0, 1, 2, 3
        
        for i in range(4):
            for k in range(7):
                U[a_row_base + i, base_col + k] += u_2x2[i][k]
                V[b_row_base + i, base_col + k] += v_2x2[i][k]
                W[c_row_base + i, base_col + k] += w_2x2[i][k]
                
    # Helper to add 4x4x4 Strassen (using 2x2x2 Strassen)
    # A is 4x4, B is 4x4, C is 4x4
    # Split into 2x2 blocks
    def add_strassen_4x4(a_row_base, b_row_base, c_row_base, base_col):
        # Strassen 4x4x4 using 2x2 blocks
        # We need 7 multiplications of 2x2 matrices.
        # Each 2x2 multiplication is rank 7.
        # Total rank 49.
        
        # Block indices for 4x4 matrix:
        # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1) in block coords
        # Global row index for block (bi, bj) is bi*2 + bj? No.
        # A is 4x4. Block (bi, bj) is A[2bi:2bi+2, 2bj:2bj+2].
        # The "row" in U corresponds to a specific element of A.
        # U index: i*m + j. Here m=4.
        # So A[i, j] is at U row i*4 + j.
        
        # Let's define the 7 sub-multiplications.
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # For each Mk, we have an Xk (2x2) and Yk (2x2).
        # Xk is formed from A blocks. Yk from B blocks.
        # We need to add the contribution of Xk * Yk to C blocks.
        
        # We will call add_strassen_2x2 for each of the 7 sub-problems.
        # But add_strassen_2x2 expects contiguous row bases for the 2x2 blocks.
        # The 2x2 blocks of A are NOT contiguous in the U matrix (which is 16xR).
        # U is flattened.
        # A[0,0] is U[0], A[0,1] is U[1], A[0,2] is U[2], A[0,3] is U[3]
        # A[1,0] is U[4], A[1,1] is U[5] ...
        # Block A00 (top-left 2x2) consists of U[0], U[1], U[4], U[5].
        # This is not contiguous.
        
        # So I cannot directly use add_strassen_2x2 with simple base offsets.
        # I need to manually map the 16 elements of A to the 4 "virtual" rows of the 2x2 block.
        
        # Let's refactor add_strassen_2x2 to accept explicit row mappings.
        
        # Actually, it's easier to just write the 49 terms directly or use a helper that takes mappings.
        
        # Mapping for A blocks to global U rows:
        # A00: [0, 1, 4, 5]  (rows 0,1 and 4,5)
        # A01: [2, 3, 6, 7]
        # A10: [8, 9, 12, 13]
        # A11: [10, 11, 14, 15]
        
        # Mapping for B blocks to global V rows:
        # B00: [0, 1, 4, 5]
        # B01: [2, 3, 6, 7]
        # B10: [8, 9, 12, 13]
        # B11: [10, 11, 14, 15]
        
        # Mapping for C blocks to global W rows:
        # C00: [0, 1, 4, 5]
        # C01: [2, 3, 6, 7]
        # C10: [8, 9, 12, 13]
        # C11: [10, 11, 14, 15]
        
        a_maps = {
            0: [0, 1, 4, 5],
            1: [2, 3, 6, 7],
            2: [8, 9, 12, 13],
            3: [10, 11, 14, 15]
        }
        b_maps = {
            0: [0, 1, 4, 5],
            1: [2, 3, 6, 7],
            2: [8, 9, 12, 13],
            3: [10, 11, 14, 15]
        }
        c_maps = {
            0: [0, 1, 4, 5],
            1: [2, 3, 6, 7],
            2: [8, 9, 12, 13],
            3: [10, 11, 14, 15]
        }
        
        # Define the 7 sub-problems
        # Each sub-problem is (A_blocks, B_blocks, C_blocks, coeffs)
        # A_blocks: list of (block_id, coeff)
        # B_blocks: list of (block_id, coeff)
        # C_blocks: list of (block_id, coeff)
        
        # M1 = (A00+A11)(B00+B11)
        # X1 = A00 + A11. Y1 = B00 + B11.
        # Z1 = X1 Y1.
        # C00 += Z1, C11 += Z1? No.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to add Z1 to C00 and C11.
        # So for M1:
        # A part: A00 (coeff 1), A11 (coeff 1)
        # B part: B00 (coeff 1), B11 (coeff 1)
        # C part: C00 (coeff 1), C11 (coeff 1) -- Wait, Z1 is a 2x2 matrix.
        # Z1[0,0] goes to C00[0,0], Z1[0,1] to C00[0,1] etc?
        # No, Z1 is the result of multiplying two 2x2 matrices.
        # Z1 is a 2x2 block.
        # It contributes to the corresponding 2x2 block of C.
        # M1 contributes to C00 and C11?
        # Formula:
        # C00 = M1 + M4 - M5 + M7
        # C11 = M1 - M2 + M3 + M6
        # So M1 (a 2x2 matrix) is added to C00 (2x2) and C11 (2x2).
        
        # So for each Mk, we have:
        # A_blocks: dict {block_id: coeff}
        # B_blocks: dict {block_id: coeff}
        # C_contributions: list of (block_id, coeff)
        
        subs = [
            # M1
            (
                {0: 1, 3: 1}, # A00 + A11
                {0: 1, 3: 1}, # B00 + B11
                [(0, 1), (3, 1)] # C00 += M1, C11 += M1
            ),
            # M2
            (
                {2: 1, 3: 1}, # A10 + A11
                {0: 1},       # B00
                [(2, 1), (3, -1)] # C10 += M2, C11 -= M2
            ),
            # M3
            (
                {0: 1},       # A00
                {1: 1, 3: -1},# B01 - B11
                [(1, 1), (3, 1)] # C01 += M3, C11 += M3
            ),
            # M4
            (
                {3: 1},       # A11
                {2: 1, 0: -1} # B10 - B00
                ,[(0, 1), (2, 1)] # C00 += M4, C10 += M4
            ),
            # M5
            (
                {0: 1, 1: 1}, # A00 + A01
                {3: 1},       # B11
                [(0, -1), (1, 1)] # C00 -= M5, C01 += M5
            ),
            # M6
            (
                {2: 1, 0: -1}, # A10 - A00
                {0: 1, 1: 1},  # B00 + B01
                [(3, 1)]       # C11 += M6
            ),
            # M7
            (
                {1: 1, 3: -1}, # A01 - A11
                {2: 1, 3: 1},  # B10 + B11
                [(0, 1)]       # C00 += M7
            )
        ]
        
        # For each sub-problem, we decompose the 2x2x2 multiplication.
        # The 2x2x2 multiplication has rank 7.
        # We need to map the "virtual" 2x2 rows to the global rows.
        
        for a_blocks, b_blocks, c_contribs in subs:
            # We need to construct the U, V, W contributions for this 2x2x2 block.
            # The "rows" of the virtual 2x2 A are 0,1,2,3.
            # Row 0 corresponds to A[0,0] of the block.
            # In global A, this is a sum of elements from the mapped blocks.
            
            # Let's create local U, V, W for the 2x2x2 part (size 4x7)
            # Then distribute them to global U, V, W.
            
            # Local U (4x7) for Strassen 2x2x2
            u_loc = [
                [1, 0, 1, 0, 1, -1, 0],
                [0, 0, 0, 0, 1, 0, 1],
                [0, 1, 0, 0, 0, 1, 0],
                [1, 1, 0, 1, 0, 0, -1]
            ]
            v_loc = [
                [1, 1, 0, -1, 0, 1, 0],
                [0, 0, 1, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 1],
                [1, 0, -1, 0, 1, 0, 1]
            ]
            w_loc = [
                [1, 0, 0, 1, -1, 0, 1],
                [0, 0, 1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0, 0, 0],
                [1, -1, 1, 0, 0, 1, 0]
            ]
            
            # For each of the 7 terms in the 2x2x2 decomposition:
            for k in range(7):
                # Global column for this term
                col = base_col + k
                
                # Accumulate U contributions
                # For each local row i (0..3)
                for i in range(4):
                    u_val = u_loc[i][k]
                    if u_val == 0: continue
                    # This local row i corresponds to a linear combination of global A elements.
                    # The local 2x2 matrix X has elements X[0,0], X[0,1], X[1,0], X[1,1].
                    # i=0 -> X[0,0], i=1 -> X[0,1], i=2 -> X[1,0], i=3 -> X[1,1].
                    # X is formed from A blocks.
                    # X[0,0] = sum(coeff * A_block[0,0])
                    # X[0,1] = sum(coeff * A_block[0,1])
                    # X[1,0] = sum(coeff * A_block[1,0])
                    # X[1,1] = sum(coeff * A_block[1,1])
                    
                    # We need to map local index i to (block_row, block_col) in the 2x2 block.
                    if i == 0: br, bc = 0, 0
                    elif i == 1: br, bc = 0, 1
                    elif i == 2: br, bc = 1, 0
                    else: br, bc = 1, 1
                    
                    # For each A block involved in X
                    for block_id, coeff in a_blocks.items():
                        # The element at (br, bc) in this block is at global row:
                        # block_map[block_id] gives [r00, r01, r10, r11]
                        # Index br*2+bc? No.
                        # My maps: [0, 1, 4, 5] for block 0.
                        # 0: (0,0), 1: (0,1), 4: (1,0), 5: (1,1).
                        # So index is br*2 + bc?
                        # br=0, bc=0 -> 0. br=0, bc=1 -> 1. br=1, bc=0 -> 4. br=1, bc=1 -> 5.
                        # Yes.
                        global_row = a_maps[block_id][br*2+bc]
                        U[global_row, col] += u_val * coeff
                        
                # Accumulate V contributions
                for i in range(4):
                    v_val = v_loc[i][k]
                    if v_val == 0: continue
                    if i == 0: br, bc = 0, 0
                    elif i == 1: br, bc = 0, 1
                    elif i == 2: br, bc = 1, 0
                    else: br, bc = 1, 1
                    
                    for block_id, coeff in b_blocks.items():
                        global_row = b_maps[block_id][br*2+bc]
                        V[global_row, col] += v_val * coeff
                        
                # Accumulate W contributions
                # W is the result.
                # For each C block involved in the contribution
                for block_id, coeff in c_contribs:
                    # The 2x2 result Z is added to C_block with coeff.
                    # Z[0,0] -> C_block[0,0], etc.
                    for i in range(4):
                        w_val = w_loc[i][k]
                        if w_val == 0: continue
                        if i == 0: br, bc = 0, 0
                        elif i == 1: br, bc = 0, 1
                        elif i == 2: br, bc = 1, 0
                        else: br, bc = 1, 1
                        
                        global_row = c_maps[block_id][br*2+bc]
                        W[global_row, col] += w_val * coeff
                        
    # Part 1: A * B1 (cols 0-3 of B)
    # B1 is 4x4.
    # V rows for B1: 0-15 (since m=4, p=9, B is 4x9. B1 uses cols 0-3).
    # V index: j*p + k. j=0..3, k=0..3.
    # Rows 0..3 (j=0), 9..12 (j=1), 18..21 (j=2), 27..30 (j=3).
    # Wait, V is (m*p) x rank. m=4, p=9. Size 36 x R.
    # B1 is the first 4 columns.
    # B[j, k] for k=0..3.
    # V row index: j*9 + k.
    # So B1 rows are:
    # j=0: 0,1,2,3
    # j=1: 9,10,11,12
    # j=2: 18,19,20,21
    # j=3: 27,28,29,30
    
    b1_maps = {
        0: [0, 1, 9, 10],   # Block 00: (0,0),(0,1),(1,0),(1,1) -> rows 0,1,9,10
        1: [2, 3, 11, 12],  # Block 01
        2: [18, 19, 27, 28],# Block 10
        3: [20, 21, 29, 30] # Block 11
    }
    
    # W rows for C1 (cols 0-3 of C)
    # C is 4x9. C1 is first 4 cols.
    # W index: k*n + i. k=0..3 (col), i=0..3 (row).
    # Rows:
    # k=0: 0,1,2,3
    # k=1: 4,5,6,7
    # k=2: 8,9,10,11
    # k=3: 12,13,14,15
    
    c1_maps = {
        0: [0, 1, 4, 5],   # Block 00
        1: [2, 3, 6, 7],   # Block 01
        2: [8, 9, 12, 13], # Block 10
        3: [10, 11, 14, 15]# Block 11
    }
    
    # A rows are always 0-15.
    a_maps_global = {
        0: [0, 1, 4, 5],
        1: [2, 3, 6, 7],
        2: [8, 9, 12, 13],
        3: [10, 11, 14, 15]
    }
    
    # We need to modify add_strassen_4x4 to accept custom maps.
    # For now, I'll just call it with the standard maps for A, and override B and C maps.
    
    # Let's just implement the logic inline for Part 1, 2, 3 to be safe.
    
    # Part 1: A * B1
    # Use add_strassen_4x4 logic with b1_maps and c1_maps.
    # I will copy the add_strassen_4x4 code and adapt it.
    
    def add_strassen_4x4_custom(a_maps, b_maps, c_maps, base_col):
        subs = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, [(0, 1), (3, 1)]),
            ({2: 1, 3: 1}, {0: 1}, [(2, 1), (3, -1)]),
            ({0: 1}, {1: 1, 3: -1}, [(1, 1), (3, 1)]),
            ({3: 1}, {2: 1, 0: -1}, [(0, 1), (2, 1)]),
            ({0: 1, 1: 1}, {3: 1}, [(0, -1), (1, 1)]),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, [(3, 1)]),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, [(0, 1)])
        ]
        
        u_loc = [
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1]
        ]
        v_loc = [
            [1, 1, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [1, 0, -1, 0, 1, 0, 1]
        ]
        w_loc = [
            [1, 0, 0, 1, -1, 0, 1],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ]
        
        for a_blocks, b_blocks, c_contribs in subs:
            for k in range(7):
                col = base_col + k
                for i in range(4):
                    u_val = u_loc[i][k]
                    if u_val == 0: continue
                    br, bc = (0,0) if i==0 else (0,1) if i==1 else (1,0) if i==2 else (1,1)
                    for block_id, coeff in a_blocks.items():
                        global_row = a_maps[block_id][br*2+bc]
                        U[global_row, col] += u_val * coeff
                        
                    v_val = v_loc[i][k]
                    if v_val == 0: continue
                    for block_id, coeff in b_blocks.items():
                        global_row = b_maps[block_id][br*2+bc]
                        V[global_row, col] += v_val * coeff
                        
                    w_val = w_loc[i][k]
                    if w_val == 0: continue
                    for block_id, coeff in c_contribs:
                        global_row = c_maps[block_id][br*2+bc]
                        W[global_row, col] += w_val * coeff

    add_strassen_4x4_custom(a_maps_global, b1_maps, c1_maps, 0)
    
    # Part 2: A * B2 (cols 4-7 of B)
    # B2 rows:
    # j=0: 4,5,6,7
    # j=1: 13,14,15,16
    # j=2: 22,23,24,25
    # j=3: 31,32,33,34
    
    b2_maps = {
        0: [4, 5, 13, 14],
        1: [6, 7, 15, 16],
        2: [22, 23, 31, 32],
        3: [24, 25, 33, 34]
    }
    
    # C2 rows (cols 4-7 of C)
    # k=4: 16,17,18,19
    # k=5: 20,21,22,23
    # k=6: 24,25,26,27
    # k=7: 28,29,30,31
    
    c2_maps = {
        0: [16, 17, 20, 21],
        1: [18, 19, 22, 23],
        2: [24, 25, 28, 29],
        3: [26, 27, 30, 31]
    }
    
    add_strassen_4x4_custom(a_maps_global, b2_maps, c2_maps, 49)
    
    # Part 3: A * B3 (col 8 of B)
    # B3 is 4x1.
    # V rows: j*9 + 8.
    # j=0: 8
    # j=1: 17
    # j=2: 26
    # j=3: 35
    
    # C3 is 4x1.
    # W rows: 8*4 + i = 32+i.
    # i=0: 32
    # i=1: 33
    # i=2: 34
    # i=3: 35
    
    # Schoolbook for 4x4x1
    # Rank 16.
    base_col = 98
    for i in range(4):
        for j in range(4):
            col = base_col + i*4 + j
            U[i*4 + j, col] = 1
            V[j*9 + 8, col] = 1
            W[32 + i, col] = 1
            
    return {"U": U, "V": V, "W": W}
