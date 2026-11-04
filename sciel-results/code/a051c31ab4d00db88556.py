import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 49 + 28 = 77
    rank = 77
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 steps: each step is (alpha, beta, gamma) where each is a dict from (i,j) to coeff
    # A is 2x2, B is 2x2, C = A*B is 2x2
    # C_00 = M1 - M4 - M6 + M7
    # C_01 = M2 + M4
    # C_10 = M3 + M5
    # C_11 = M1 - M3 - M5 + M6
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    strassen_steps_222 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),       # M1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): 1}),                  # M2
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),                 # M3
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),                 # M4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),                 # M5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),                 # M6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})                  # M7
    ]
    
    # Part 1: 4x4x4 part (cols 0-48)
    # A is 4x4, B1 is 4x4 (cols 0-3 of B), C1 is 4x4 (cols 0-3 of C)
    # Use Strassen at 4x4 level: 7 multiplications of 2x2x2
    # Each 2x2x2 multiplication uses Strassen: 7 multiplications of 1x1x1
    # Total: 7 * 7 = 49
    
    col_start_1 = 0
    
    # For each of the 7 Strassen steps for the 4x4 level
    for s in range(7):
        alpha_4, beta_4, gamma_4 = strassen_steps_222[s]
        # For each of the 7 Strassen steps for the 2x2 level (inside each block multiplication)
        for s2 in range(7):
            alpha_2, beta_2, gamma_2 = strassen_steps_222[s2]
            
            col = col_start_1 + s * 7 + s2
            
            # Process U (A)
            for (bi, bj), val_a4 in alpha_4.items():
                # bi, bj are block indices in 4x4 A (0 or 1)
                # The block is 2x2.
                # Inside this block, we have 2x2 sub-blocks (0,0), (0,1), (1,0), (1,1)
                for (si, sj), val_a2 in alpha_2.items():
                    # Global row index in A
                    # Block (bi, bj) starts at row bi*2, col bj*2
                    # Sub-block (si, sj) starts at row si, col sj within the block
                    # So global row = bi*2 + si, global col = bj*2 + sj
                    r = (bi * 2 + si) * m + (bj * 2 + sj)
                    U[r][col] += val_a4 * val_a2
                    
            # Process V (B1)
            for (bi, bj), val_b4 in beta_4.items():
                for (si, sj), val_b2 in beta_2.items():
                    # B1 is 4x4.
                    # Block (bi, bj) starts at row bi*2, col bj*2
                    # Sub-block (si, sj) starts at row si, col sj
                    # Global row = bi*2 + si, global col = bj*2 + sj
                    # Note: V index is row * p + col.
                    # But B1 is only the first 4 columns of B.
                    # So col index in B is bj*2 + sj.
                    r = (bi * 2 + si) * p + (bj * 2 + sj)
                    V[r][col] += val_b4 * val_b2
                    
            # Process W (C1)
            for (bi, bj), val_c4 in gamma_4.items():
                for (si, sj), val_c2 in gamma_2.items():
                    # C1 is 4x4.
                    # Block (bi, bj) starts at row bi*2, col bj*2
                    # Sub-block (si, sj) starts at row si, col sj
                    # Global row = bi*2 + si, global col = bj*2 + sj
                    # W index is col * n + row.
                    r = (bj * 2 + sj) * n + (bi * 2 + si)
                    W[r][col] += val_c4 * val_c2

    # Part 2: 4x4x2 part (cols 49-76)
    # A is 4x4, B2 is 4x2 (cols 4-5 of B), C2 is 4x2 (cols 4-5 of C)
    # Partition A into 2x2 blocks of size 2x2.
    # Partition B2 into 2 blocks of size 2x2 (top and bottom).
    # Partition C2 into 2 blocks of size 2x2 (top and bottom).
    
    col_start_2 = 49
    
    # We need 4 multiplications:
    # M1: A00 * B0
    # M2: A01 * B1
    # M3: A10 * B0
    # M4: A11 * B1
    
    # Each multiplication is 2x2x2, rank 7.
    # Total 28 columns.
    
    mults = [
        ((0,0), (0,0)), # A00, B0 -> C0
        ((0,1), (1,0)), # A01, B1 -> C0
        ((1,0), (0,0)), # A10, B0 -> C1
        ((1,1), (1,0))  # A11, B1 -> C1
    ]
    
    # The mapping of B and C blocks:
    # B2 is 4x2.
    # B0 is rows 0-1, cols 0-1 (relative to B2, so global cols 4-5).
    # B1 is rows 2-3, cols 0-1 (relative to B2, so global cols 4-5).
    # C2 is 4x2.
    # C0 is rows 0-1, cols 0-1 (relative to C2, so global cols 4-5).
    # C1 is rows 2-3, cols 0-1 (relative to C2, so global cols 4-5).
    
    # A blocks:
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    # For each of the 4 multiplications:
    for m_idx, (a_block, b_block) in enumerate(mults):
        # a_block: (bi, bj) in A
        # b_block: (bi, bj) in B2? No, B2 is split vertically.
        # Let's define b_block as (bi, 0) where bi is 0 or 1.
        # And c_block is determined by a_block's bi?
        # C0 comes from A00, A01 (bi=0).
        # C1 comes from A10, A11 (bi=1).
        # So c_block bi is same as a_block bi.
        
        c_block = (a_block[0], 0) # bi same, bj=0 (since C2 is 2x2 blocks? No, C2 is 4x2, split into two 2x2 blocks vertically).
        # Actually, C2 is 4x2.
        # C0 (top) is rows 0-1, cols 0-1.
        # C1 (bottom) is rows 2-3, cols 0-1.
        
        # For each of the 7 Strassen steps inside this 2x2x2 multiplication:
        for s2 in range(7):
            alpha_2, beta_2, gamma_2 = strassen_steps_222[s2]
            col = col_start_2 + m_idx * 7 + s2
            
            # Process U (A)
            for (si, sj), val_a2 in alpha_2.items():
                # A block is a_block (bi, bj).
                # Sub-block (si, sj).
                # Global row = a_block[0]*2 + si
                # Global col = a_block[1]*2 + sj
                r = (a_block[0] * 2 + si) * m + (a_block[1] * 2 + sj)
                U[r][col] += val_a2
                
            # Process V (B2)
            for (si, sj), val_b2 in beta_2.items():
                # B2 block is b_block (bi, 0).
                # Sub-block (si, sj).
                # Global row = b_block[0]*2 + si
                # Global col = 4 + sj (since B2 starts at col 4)
                r = (b_block[0] * 2 + si) * p + (4 + sj)
                V[r][col] += val_b2
                
            # Process W (C2)
            for (si, sj), val_c2 in gamma_2.items():
                # C2 block is c_block (bi, 0).
                # Sub-block (si, sj).
                # Global row = c_block[0]*2 + si
                # Global col = 4 + sj
                # W index is col * n + row
                r = (4 + sj) * n + (c_block[0] * 2 + si)
                W[r][col] += val_c2

    return {"U": U, "V": V, "W": W}
