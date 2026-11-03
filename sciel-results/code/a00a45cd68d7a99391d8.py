import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 88 decomposition
    # 2x M_2,4,4 (rank 28 each) + 2x M_1,4,4 (rank 16 each)
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen factors for 2x2x2
    # Each factor is (U_coeffs, V_coeffs, W_coeffs)
    # U, V, W are dicts mapping (i,j) or (i) or (j) to coefficient
    # For 2x2x2:
    # U indices: (0,0), (0,1), (1,0), (1,1)
    # V indices: (0,0), (0,1), (1,0), (1,1)
    # W indices: (0,0), (0,1), (1,0), (1,1)
    
    strassen_factors = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # Process 2x M_2,4,4 blocks (for i=0,1)
    # The matrix A is 3x4. We take rows 0,1 for the first block.
    # The matrix B is 4x8. We split into two 4x4 blocks.
    # So we have 2 blocks of M_2,4,4.
    
    for c_block in range(2):
        # c_block 0: B columns 0-3, C columns 0-3
        # c_block 1: B columns 4-7, C columns 4-7
        
        for factor in strassen_factors:
            a_coeffs, b_coeffs, c_coeffs = factor
            
            # Each Strassen factor corresponds to one "multiplication" in the 2x2 block alg.
            # But here we are doing M_2,4,4.
            # The 2x2 structure is on the output C (2 rows) and input A (2 rows) and B (split into 2x2 blocks?).
            
            # Wait, the previous logic:
            # M_2,4,4 is decomposed into 4 M_2,2,2 multiplications?
            # No, M_2,4,4 -> A(2x4), B(4x4).
            # Split A into A_L(2x2), A_R(2x2).
            # Split B into B_TL, B_TR, B_BL, B_BR (2x2).
            # C_L = A_L B_TL + A_R B_BL
            # C_R = A_L B_TR + A_R B_BR
            # This is 4 separate M_2,2,2 multiplications.
            # Rank 4 * 7 = 28.
            
            # So for each of the 4 sub-products, we apply the 7 Strassen factors.
            # Total 28 factors per M_2,4,4 block.
            
            sub_prods = [
                # (a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset)
                # A is 2x4. B is 4x4. C is 2x4.
                # Sub-prod 0: A_L (rows 0-1, cols 0-1) * B_TL (rows 0-1, cols 0-1) -> C_L (rows 0-1, cols 0-1)
                (0, 0, 0, 0, 0, 0),
                # Sub-prod 1: A_L (rows 0-1, cols 0-1) * B_TR (rows 0-1, cols 2-3) -> C_R (rows 0-1, cols 2-3)
                (0, 0, 0, 2, 0, 2),
                # Sub-prod 2: A_R (rows 0-1, cols 2-3) * B_BL (rows 2-3, cols 0-1) -> C_L (rows 0-1, cols 0-1)
                (0, 2, 2, 0, 0, 0),
                # Sub-prod 3: A_R (rows 0-1, cols 2-3) * B_BR (rows 2-3, cols 2-3) -> C_R (rows 0-1, cols 2-3)
                (0, 2, 2, 2, 0, 2),
            ]
            
            for a_i_off, a_j_off, b_i_off, b_j_off, c_i_off, c_j_off in sub_prods:
                for (ai, aj), au in a_coeffs.items():
                    u_idx = (ai + a_i_off) * 4 + (aj + a_j_off)
                    U[u_idx][col] += au
                
                for (bi, bj), bv in b_coeffs.items():
                    # B is 4x8.
                    # b_i_off is row offset in 4x4 block.
                    # b_j_off is col offset in 4x4 block.
                    # c_block determines which 4x4 block of B (and C).
                    v_idx = (bi + b_i_off) * 8 + (bj + b_j_off + 4 * c_block)
                    V[v_idx][col] += bv
                
                for (ci, cj), cw in c_coeffs.items():
                    # C is 8x3 (p*n).
                    # w_idx = col * n + row?
                    # Convention: W[k*n+i, r] where k is col index of C, i is row index of C.
                    # C is 2x4 (for this block).
                    # c_i is row in 2x2 block (0..1).
                    # c_j is col in 2x2 block (0..1).
                    # c_i_off, c_j_off are offsets in the 2x4 block.
                    # c_block is offset in the 8x3 matrix?
                    # C total is 8 rows (p) x 3 cols (n).
                    # The 2x4 block corresponds to C rows 0-1 (if c_block=0) or 4-7 (if c_block=1)?
                    # No, C is 8x3. The "rows" of C are the output of B's columns?
                    # Let's check the convention: W[k*n+i, r].
                    # k is index of B column (0..7). i is index of A row (0..2).
                    # So W is indexed by (k, i).
                    # In the M_2,4,4 block:
                    # A rows are 0,1. So i is 0,1.
                    # B cols are 0..3 (if c_block=0) or 4..7 (if c_block=1).
                    # So k is 0..3 or 4..7.
                    # In the 2x2 sub-block:
                    # c_i is row in C (which is A row index). So c_i is 0,1.
                    # c_j is col in C (which is B col index within the 4x4 block).
                    # So global k = c_j + c_j_off + 4 * c_block.
                    # global i = c_i + c_i_off (which is just c_i since c_i_off=0).
                    
                    w_idx = (c_j + c_j_off + 4 * c_block) * 3 + (c_i + c_i_off)
                    W[w_idx][col] += cw
                
                col += 1
    
    # Process 2x M_1,4,4 blocks (for i=2)
    # A row 2 (1x4). B is 4x4. C is 1x4.
    # Schoolbook rank 16.
    
    for c_block in range(2):
        for k in range(4): # B row / A col
            for j in range(4): # B col / C col
                u_idx = 2 * 4 + k # A row 2, col k
                v_idx = k * 8 + (j + 4 * c_block) # B row k, col j + offset
                w_idx = (j + 4 * c_block) * 3 + 2 # C col j + offset, row 2
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
