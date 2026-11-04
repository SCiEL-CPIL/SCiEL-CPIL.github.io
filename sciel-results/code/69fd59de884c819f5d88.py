import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4, Rank 16 for 4x4x1
    rank_444 = 49
    rank_441 = 16
    total_rank = rank_444 + rank_441
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # 1. Fill the 4x4x4 part (columns 0-3 of B, rows 0-3 of C)
    # We need the Strassen decomposition for 4x4x4.
    # This consists of 7 Strassen multiplications of 2x2 matrices.
    # Each 2x2x2 multiplication has rank 7.
    
    # Helper to get Strassen 2x2x2 U, V, W
    # Indices for 2x2: A is 2x2, B is 2x2, C is 2x2.
    # U shape (4, 7), V shape (4, 7), W shape (4, 7).
    
    def get_strassen_222():
        u = [[0]*7 for _ in range(4)]
        v = [[0]*7 for _ in range(4)]
        w = [[0]*7 for _ in range(4)]
        
        # M1 = (A11+A22)(B11+B22)
        # A11 is index 0, A22 is index 3
        # B11 is index 0, B22 is index 3
        # C11 is index 0
        u[0][0] = 1; u[3][0] = 1
        v[0][0] = 1; v[3][0] = 1
        w[0][0] = 1; w[1][0] = 1; w[2][0] = 1; w[3][0] = 1 # C11, C12, C21, C22?
        # Wait, C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's map the 7 multiplications to columns 0-6.
        
        # M1
        u[0][0] = 1; u[3][0] = 1
        v[0][0] = 1; v[3][0] = 1
        w[0][0] = 1  # C11
        w[3][0] = 1  # C22
        
        # M2 = (A21+A22)B11
        u[2][1] = 1; u[3][1] = 1
        v[0][1] = 1
        w[2][1] = 1  # C21
        w[3][1] = -1 # C22
        
        # M3 = A11(B12-B22)
        u[0][2] = 1
        v[1][2] = 1; v[3][2] = -1
        w[1][2] = 1  # C12
        w[3][2] = 1  # C22
        
        # M4 = A22(B21-B11)
        u[3][3] = 1
        v[2][3] = 1; v[0][3] = -1
        w[0][3] = 1  # C11
        w[2][3] = 1  # C21
        
        # M5 = (A11+A12)B22
        u[0][4] = 1; u[1][4] = 1
        v[3][4] = 1
        w[0][4] = -1 # C11
        w[1][4] = 1  # C12
        
        # M6 = (A21-A11)(B11+B12)
        u[2][5] = 1; u[0][5] = -1
        v[0][5] = 1; v[1][5] = 1
        w[3][5] = 1  # C22
        
        # M7 = (A12-A22)(B21+B22)
        u[1][6] = 1; u[3][6] = -1
        v[2][6] = 1; v[3][6] = 1
        w[0][6] = 1  # C11
        
        return u, v, w

    u222, v222, w222 = get_strassen_222()
    
    # Now map this to 4x4x4.
    # The 4x4 matrices are composed of 2x2 blocks.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # Block indices:
    # A11: rows 0-1, cols 0-1 -> indices 0,1,4,5 in flat 4x4?
    # Flat index for 4x4: i*4 + j.
    # A11 (2x2):
    #   (0,0)->0, (0,1)->1
    #   (1,0)->4, (1,1)->5
    # A12 (2x2):
    #   (0,2)->2, (0,3)->3
    #   (1,2)->6, (1,3)->7
    # A21 (2x2):
    #   (2,0)->8, (2,1)->9
    #   (3,0)->12, (3,1)->13
    # A22 (2x2):
    #   (2,2)->10, (2,3)->11
    #   (3,2)->14, (3,3)->15
    
    # B is 4x4 (first 4 cols).
    # B11: rows 0-1, cols 0-1 -> indices 0,1,4,5
    # B12: rows 0-1, cols 2-3 -> indices 2,3,6,7
    # B21: rows 2-3, cols 0-1 -> indices 8,9,12,13
    # B22: rows 2-3, cols 2-3 -> indices 10,11,14,15
    
    # C is 4x4.
    # C11: rows 0-1, cols 0-1 -> indices 0,1,4,5
    # C12: rows 0-1, cols 2-3 -> indices 2,3,6,7
    # C21: rows 2-3, cols 0-1 -> indices 8,9,12,13
    # C22: rows 2-3, cols 2-3 -> indices 10,11,14,15
    
    # We have 7 Strassen multiplications.
    # Each uses the 2x2x2 decomposition.
    # We need to map the 2x2 indices to 4x4 indices.
    
    # Mapping for A (U):
    # 2x2 index 0 (A11) -> 4x4 indices [0, 4] (col 0 of A11) ? No.
    # In 2x2, index is i*2+j.
    # 0: (0,0)
    # 1: (0,1)
    # 2: (1,0)
    # 3: (1,1)
    
    # In 4x4, A11 is at (0,0)..(1,1).
    # So 2x2 index 0 maps to 4x4 index 0.
    # 2x2 index 1 maps to 4x4 index 1.
    # 2x2 index 2 maps to 4x4 index 4.
    # 2x2 index 3 maps to 4x4 index 5.
    
    # For A12 (top right block of A):
    # 2x2 index 0 maps to 4x4 index 2.
    # 2x2 index 1 maps to 4x4 index 3.
    # 2x2 index 2 maps to 4x4 index 6.
    # 2x2 index 3 maps to 4x4 index 7.
    
    # General mapping for A block (br, bc) where br, bc in {0,1}:
    # base_row = br * 2
    # base_col = bc * 2
    # 2x2 index k (i*2+j) maps to 4x4 index (base_row + i)*4 + (base_col + j).
    
    # We have 7 multiplications.
    # For each multiplication, we determine which A blocks and B blocks are involved.
    # M1: A11+A22, B11+B22
    # M2: A21+A22, B11
    # M3: A11, B12-B22
    # M4: A22, B21-B11
    # M5: A11+A12, B22
    # M6: A21-A11, B11+B12
    # M7: A12-A22, B21+B22
    
    # We will iterate over the 7 multiplications.
    # For each, we use the u222, v222, w222 templates.
    # We need to add the contributions to U, V, W.
    
    # U has 16 rows.
    # V has 20 rows (4x5). We only use first 16 (4x4) for this part.
    # W has 20 rows (5x4). We only use first 16 (4x4) for this part.
    
    for mult_idx in range(7):
        # Determine A blocks and coefficients
        # u222 has 4 rows (2x2 indices).
        # We need to map each 2x2 index to the corresponding 4x4 indices for the involved blocks.
        
        # This is tricky because the blocks involved depend on the multiplication.
        # Let's hardcode the block selections for each of the 7 multiplications.
        
        # M1: A11, A22. B11, B22.
        # M2: A21, A22. B11.
        # M3: A11. B12, B22.
        # M4: A22. B21, B11.
        # M5: A11, A12. B22.
        # M6: A21, A11. B11, B12.
        # M7: A12, A22. B21, B22.
        
        # We can define for each multiplication:
        # a_blocks: list of (block_row, block_col, coeff)
        # b_blocks: list of (block_row, block_col, coeff)
        # c_blocks: list of (block_row, block_col, coeff)
        
        # Block indices: (0,0)=A11, (0,1)=A12, (1,0)=A21, (1,1)=A22
        
        a_specs = [
            [(0,0,1), (1,1,1)], # M1
            [(1,0,1), (1,1,1)], # M2
            [(0,0,1)],          # M3
            [(1,1,1)],          # M4
            [(0,0,1), (0,1,1)], # M5
            [(1,0,1), (0,0,-1)],# M6
            [(0,1,1), (1,1,-1)] # M7
        ]
        
        b_specs = [
            [(0,0,1), (1,1,1)], # M1
            [(0,0,1)],          # M2
            [(0,1,1), (1,1,-1)],# M3
            [(1,0,1), (0,0,-1)],# M4
            [(1,1,1)],          # M5
            [(0,0,1), (0,1,1)], # M6
            [(1,0,1), (1,1,1)]  # M7
        ]
        
        c_specs = [
            [(0,0,1), (1,1,1)], # M1
            [(1,0,1), (1,1,-1)],# M2
            [(0,1,1), (1,1,1)], # M3
            [(0,0,1), (1,0,1)], # M4
            [(0,0,-1), (0,1,1)],# M5
            [(1,1,1)],          # M6
            [(0,0,1)]           # M7
        ]
        
        # Now, for each 2x2 index in u222, we add to U.
        # u222[i][mult_idx] is the coefficient for 2x2 index i in this multiplication.
        # But u222 is the same for all 7? No, u222 is the template for ONE 2x2x2 multiplication.
        # The 7 multiplications are distinct instances.
        # So for mult_idx, we use column mult_idx of u222, v222, w222.
        
        for i22 in range(4):
            coeff_u = u222[i22][mult_idx]
            if coeff_u == 0: continue
            
            # Map i22 to 4x4 indices for the specified A blocks.
            # i22 is 0..3.
            # r22 = i22 // 2
            # c22 = i22 % 2
            
            # We need to add coeff_u * (contribution from each block) to U.
            # But the A blocks are summed with coefficients in a_specs.
            # So for each block (br, bc, coeff_a) in a_specs[mult_idx]:
            # The 2x2 index i22 corresponds to a specific element in that block.
            # That element is at 4x4 index (br*2 + r22)*4 + (bc*2 + c22).
            # We add coeff_u * coeff_a to U[that_index][rank_offset].
            
            r22 = i22 // 2
            c22 = i22 % 2
            
            for br, bc, coeff_a in a_specs[mult_idx]:
                idx_44 = (br * 2 + r22) * 4 + (bc * 2 + c22)
                U[idx_44][mult_idx] += coeff_u * coeff_a
        
        for j22 in range(4):
            coeff_v = v222[j22][mult_idx]
            if coeff_v == 0: continue
            
            r22 = j22 // 2
            c22 = j22 % 2
            
            for br, bc, coeff_b in b_specs[mult_idx]:
                idx_44 = (br * 2 + r22) * 4 + (bc * 2 + c22)
                V[idx_44][mult_idx] += coeff_v * coeff_b
        
        for k22 in range(4):
            coeff_w = w222[k22][mult_idx]
            if coeff_w == 0: continue
            
            r22 = k22 // 2
            c22 = k22 % 2
            
            for br, bc, coeff_c in c_specs[mult_idx]:
                idx_44 = (br * 2 + r22) * 4 + (bc * 2 + c22)
                W[idx_44][mult_idx] += coeff_w * coeff_c

    # 2. Fill the 4x4x1 part (column 4 of B, all rows of C)
    # Naive: 16 multiplications.
    # A is 4x4, B is 4x1 (column 4 of B).
    # C is 4x1 (column 4 of C).
    # C_i = sum_j A_ij B_j.
    # Rank 16.
    
    start_rank = rank_444
    for i in range(4):
        for j in range(4):
            col = start_rank + i * 4 + j
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1 # B is 4x5, so index is j*5 + 4
            W[4 * 4 + i][col] = 1 # C is 5x4, so index is 4*4 + i (row 4, col i)
            
    return {"U": U, "V": V, "W": W}
