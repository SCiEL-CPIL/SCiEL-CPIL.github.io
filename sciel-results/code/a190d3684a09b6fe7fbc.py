"""Matrix multiplication tensor decomposition for 4x5 * 5x6.
Uses Strassen's algorithm on a 4x4x4 sub-block to reduce rank.
"""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    import numpy as np
    
    # We'll use a block decomposition approach.
    # Split the middle dimension (5) into 4 + 1.
    # A = [A1 | A2] where A1 is 4x4, A2 is 4x1
    # B = [B1; B2] where B1 is 4x6, B2 is 1x6
    # C = A1*B1 + A2*B2
    
    # For A1*B1 (4x4 * 4x6):
    #   Split B1 into two 4x3 blocks: B1 = [B1a | B1b]
    #   A1*B1a: 4x4 * 4x3 -> use Strassen on 4x4x4 part + schoolbook for remaining
    #   Actually, let's use a cleaner approach.
    
    # Let's use Strassen for 4x4x4 multiplication.
    # The 4x4x4 tensor can be decomposed with rank 49 using Strassen recursion.
    
    # Strassen 2x2x2 has 7 terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 4x4x4, we apply Strassen recursively:
    # Split 4x4 into 2x2 blocks.
    # A = [[A00, A01], [A10, A11]] where each block is 2x2
    # B = [[B00, B01], [B10, B11]] where each block is 2x2
    # C = A*B = [[C00, C01], [C10, C11]] where each block is 2x2
    
    # Each Ci,j is a 2x2x2 multiplication of the corresponding blocks.
    # Using Strassen, we compute 7 intermediate 2x2x2 products.
    # Each 2x2x2 product uses 7 scalar multiplications.
    # Total: 7 * 7 = 49 scalar multiplications.
    
    # Let's construct the 49-term decomposition for 4x4x4.
    
    # Define the 7 Strassen terms for 2x2x2.
    # Each term is a rank-1 tensor in the 2x2x2 space.
    # For term t, we have U_t (4 entries), V_t (4 entries), W_t (4 entries).
    
    # Indexing: A[i,j] -> index i*2+j, B[i,j] -> index i*2+j, C[i,j] -> index i*2+j
    
    # Term 1: (A00+A11)(B00+B11)
    # U: A00=1, A11=1, A01=0, A10=0 -> [1,0,0,1]
    # V: B00=1, B11=1, B01=0, B10=0 -> [1,0,0,1]
    # W: C00 += 1, C01 -= 0... 
    # C00 = M1 - M4 + M5 + M7 -> M1 contributes +1 to C00
    # C01 = M3 + M5 -> M1 contributes 0 to C01
    # C10 = M2 + M4 -> M1 contributes 0 to C10
    # C11 = M1 - M2 + M3 + M6 -> M1 contributes +1 to C11
    # W: [1, 0, 0, 1]
    
    # Term 2: (A10+A11)B00
    # U: A10=1, A11=1 -> [0,0,1,1]
    # V: B00=1 -> [1,0,0,0]
    # W: C10 += 1 -> [0,0,1,0]
    
    # Term 3: A00(B10-B11)
    # U: A00=1 -> [1,0,0,0]
    # V: B10=1, B11=-1 -> [0,0,1,-1]
    # W: C01 += 1, C11 += 1 -> [0,1,0,1]
    
    # Term 4: A11(B01-B10)
    # U: A11=1 -> [0,0,0,1]
    # V: B01=1, B10=-1 -> [0,1,-1,0]
    # W: C00 -= 1, C10 += 1 -> [-1,0,1,0]
    
    # Term 5: (A00+A10)B11
    # U: A00=1, A10=1 -> [1,0,1,0]
    # V: B11=1 -> [0,0,0,1]
    # W: C00 += 1, C01 += 1 -> [1,1,0,0]
    
    # Term 6: (A10-A00)(B00+B10)
    # U: A10=1, A00=-1 -> [-1,0,1,0]
    # V: B00=1, B10=1 -> [1,0,1,0]
    # W: C11 += 1 -> [0,0,0,1]
    
    # Term 7: (A01-A11)(B01+B11)
    # U: A01=1, A11=-1 -> [0,1,0,-1]
    # V: B01=1, B11=1 -> [0,1,0,1]
    # W: C00 += 1, C11 += 1 -> [1,0,0,1]
    
    strassen_2x2 = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        ([1,0,0,0], [0,0,1,-1], [0,1,0,1]),
        ([0,0,0,1], [0,1,-1,0], [-1,0,1,0]),
        ([1,0,1,0], [0,0,0,1], [1,1,0,0]),
        ([-1,0,1,0], [1,0,1,0], [0,0,0,1]),
        ([0,1,0,-1], [0,1,0,1], [1,0,0,1]),
    ]
    
    # Now for 4x4x4, we have 4 block positions.
    # A blocks: A00 (top-left), A01 (top-right), A10 (bottom-left), A11 (bottom-right)
    # Each block is 2x2, so in the 4x4 matrix:
    # A00: rows 0-1, cols 0-1 -> indices 0,1,2,3
    # A01: rows 0-1, cols 2-3 -> indices 4,5,6,7
    # A10: rows 2-3, cols 0-1 -> indices 8,9,10,11
    # A11: rows 2-3, cols 2-3 -> indices 12,13,14,15
    
    # Similarly for B and C.
    
    # For each of the 7 Strassen terms at the block level, we need to
    # expand each block multiplication into 7 scalar multiplications.
    
    # Block-level Strassen:
    # M1 = (A00+A11)(B00+B11) -> 2x2x2 multiplication
    # M2 = (A10+A11)(B00) -> 2x2x2 multiplication
    # M3 = (A00)(B10-B11) -> 2x2x2 multiplication
    # M4 = (A11)(B01-B10) -> 2x2x2 multiplication
    # M5 = (A00+A10)(B11) -> 2x2x2 multiplication
    # M6 = (A10-A00)(B00+B10) -> 2x2x2 multiplication
    # M7 = (A01-A11)(B01+B11) -> 2x2x2 multiplication
    
    # Each Mk is a 2x2x2 multiplication, which we decompose using the 7 Strassen terms above.
    
    # For each block-level term, we identify which A-blocks and B-blocks are involved,
    # and the coefficients.
    
    # Let's define the block-level terms:
    # Each term has: list of (A_block, coeff_A), list of (B_block, coeff_B), and W_block_coeffs
    # where W_block_coeffs maps to C-blocks.
    
    # Actually, let's just directly construct the 49 terms.
    
    # For each of the 7 block-level Strassen terms, and for each of the 7 scalar-level
    # Strassen terms, we get one rank-1 term in the 4x4x4 decomposition.
    
    # Block-level term definitions:
    # Term 1: A blocks: {00:1, 11:1}, B blocks: {00:1, 11:1}, C contribution: C00+1, C11+1
    # Term 2: A blocks: {10:1, 11:1}, B blocks: {00:1}, C contribution: C10+1
    # Term 3: A blocks: {00:1}, B blocks: {10:1, 11:-1}, C contribution: C01+1, C11+1
    # Term 4: A blocks: {11:1}, B blocks: {01:1, 10:-1}, C contribution: C00-1, C10+1
    # Term 5: A blocks: {00:1, 10:1}, B blocks: {11:1}, C contribution: C00+1, C01+1
    # Term 6: A blocks: {10:1, 00:-1}, B blocks: {00:1, 10:1}, C contribution: C11+1
    # Term 7: A blocks: {01:1, 11:-1}, B blocks: {01:1, 11:1}, C contribution: C00+1, C11+1
    
    block_terms = [
        # (A_blocks: dict, B_blocks: dict, C_blocks: dict)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(0,1):1, (1,0):-1}, {(0,0):-1, (1,0):1}),
        ({(0,0):1, (1,0):1}, {(1,1):1}, {(0,0):1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}, {(0,0):1, (1,1):1}),
    ]
    
    # Block index to 4x4 index mapping:
    # Block (bi, bj) in 4x4 matrix:
    # A: row_start = bi*2, col_start = bj*2
    # For A[i,j]: i in [0,4), j in [0,4)
    # Block (bi,bj) covers rows bi*2 to bi*2+1, cols bj*2 to bj*2+1
    # Local indices (li, lj) in block -> global (bi*2+li, bj*2+lj)
    # Flat index: (bi*2+li)*4 + (bj*2+lj)
    
    def block_local_to_global(bi, bj, li, lj, dim=4):
        return (bi*2 + li) * dim + (bj*2 + lj)
    
    # Now construct the 49 terms for 4x4x4.
    terms_4x4x4 = []
    for bt in block_terms:
        A_blocks, B_blocks, C_blocks = bt
        # For each scalar-level Strassen term
        for st in strassen_2x2:
            U_local, V_local, W_local = st
            # U_local has 4 entries for the 2x2 A block
            # V_local has 4 entries for the 2x2 B block
            # W_local has 4 entries for the 2x2 C block
            
            U_4x4 = [0] * 16
            V_4x4 = [0] * 16
            W_4x4 = [0] * 16
            
            # For each A block involved
            for (bi, bj), coeff_a in A_blocks.items():
                for li in range(2):
                    for lj in range(2):
                        local_idx = li * 2 + lj
                        global_idx = block_local_to_global(bi, bj, li, lj)
                        U_4x4[global_idx] += coeff_a * U_local[local_idx]
            
            # For each B block involved
            for (bi, bj), coeff_b in B_blocks.items():
                for li in range(2):
                    for lj in range(2):
                        local_idx = li * 2 + lj
                        global_idx = block_local_to_global(bi, bj, li, lj)
                        V_4x4[global_idx] += coeff_b * V_local[local_idx]
            
            # For each C block involved
            for (bi, bj), coeff_c in C_blocks.items():
                for li in range(2):
                    for lj in range(2):
                        local_idx = li * 2 + lj
                        global_idx = block_local_to_global(bi, bj, li, lj)
                        W_4x4[global_idx] += coeff_c * W_local[local_idx]
            
            terms_4x4x4.append((U_4x4, V_4x4, W_4x4))
    
    # Now we have 49 terms for 4x4x4.
    # We need to use this for 4x5 * 5x6.
    
    # Strategy: 
    # Split middle dimension 5 into 4 + 1.
    # A is 4x5: A = [A_left (4x4) | A_right (4x1)]
    # B is 5x6: B = [B_top (4x6); B_bottom (1x6)]
    # C = A_left * B_top + A_right * B_bottom
    
    # A_left * B_top: 4x4 * 4x6
    #   Split B_top into B_top_left (4x3) and B_top_right (4x3)
    #   A_left * B_top_left: 4x4 * 4x3
    #   A_left * B_top_right: 4x4 * 4x3
    
    # For 4x4 * 4x3:
    #   We can use the 4x4x4 decomposition and ignore the last 4 columns of C.
    #   Actually, we need to be careful. The 4x4x4 decomposition computes C[i,k] for i,k in [0,4).
    #   For 4x4 * 4x3, we only need C[i,k] for k in [0,3).
    
    # Let's handle 4x4 * 4x3 directly.
    # We can use a modified Strassen approach or just use schoolbook for the 3 columns.
    # Schoolbook for 4x4 * 4x3: 4*4*3 = 48 terms.
    # But we can do better by using the 4x4x4 decomposition and truncating.
    
    # Actually, let's use a simpler approach: use the 49-term 4x4x4 decomposition
    # for the 4x4 * 4x4 part, and schoolbook for the remaining parts.
    
    # Let's restructure:
    # C = A * B where A is 4x5, B is 5x6.
    # Split B into B1 (4x6) and B2 (1x6).
    # C = A1*B1 + A2*B2 where A1 is 4x4, A2 is 4x1.
    
    # A1*B1: 4x4 * 4x6
    #   Split B1 into B1a (4x4) and B1b (4x2)
    #   A1*B1a: 4x4 * 4x4 -> use 49-term Strassen
    #   A1*B1b: 4x4 * 4x2 -> schoolbook: 4*4*2 = 32 terms
    #   Total for A1*B1: 49 + 32 = 81
    
    # A2*B2: 4x1 * 1x6 -> schoolbook: 4*1*6 = 24 terms
    
    # Total rank: 81 + 24 = 105
    
    # Let's construct this.
    
    rank = 105
    U = [[0] * rank for _ in range(n * m)]  # 4*5 = 20 rows
    V = [[0] * rank for _ in range(m * p)]  # 5*6 = 30 rows
    W = [[0] * rank for _ in range(p * n)]  # 6*4 = 24 rows
    
    col = 0
    
    # Part 1: A1*B1a (4x4 * 4x4) using 49-term Strassen
    # A1 is rows 0-3, cols 0-3 of A (which is 4x5)
    # B1a is rows 0-3, cols 0-3 of B (which is 5x6)
    # C contribution: rows 0-3, cols 0-3 of C
    
    for U_4x4, V_4x4, W_4x4 in terms_4x4x4:
        for i in range(16):
            if U_4x4[i] != 0:
                # A1 index: i in [0,16) maps to A[i//4][i%4]
                # In the full 4x5 A, this is row i//4, col i%4
                # Flat index in U: (i//4)*5 + (i%4)
                U[(i // 4) * 5 + (i % 4)][col] = U_4x4[i]
        
        for i in range(16):
            if V_4x4[i] != 0:
                # B1a index: i in [0,16) maps to B[i//4][i%4]
                # In the full 5x6 B, this is row i//4, col i%4
                # Flat index in V: (i//4)*6 + (i%4)
                V[(i // 4) * 6 + (i % 4)][col] = V_4x4[i]
        
        for i in range(16):
            if W_4x4[i] != 0:
                # C index: i in [0,16) maps to C[i//4][i%4]
                # In the full 6x4 C, this is row i%4, col i//4
                # W[k*n+i] where k is col, i is row
                # So W index: (i%4)*4 + (i//4)
                W[(i % 4) * 4 + (i // 4)][col] = W_4x4[i]
        
        col += 1
    
    # Part 2: A1*B1b (4x4 * 4x2) using schoolbook
    # A1: rows 0-3, cols 0-3 of A
    # B1b: rows 0-3, cols 4-5 of B
    # C contribution: rows 0-3, cols 4-5 of C
    
    for i in range(4):
        for j in range(4):
            for k in range(2):
                # A[i,j], B[j,4+k], C[i,4+k]
                U[i * 5 + j][col] = 1
                V[j * 6 + (4 + k)][col] = 1
                W[(4 + k) * 4 + i][col] = 1
                col += 1
    
    # Part 3: A2*B2 (4x1 * 1x6) using schoolbook
    # A2: rows 0-3, col 4 of A
    # B2: row 4, cols 0-5 of B
    # C contribution: rows 0-3, cols 0-5 of C
    
    for i in range(4):
        for k in range(6):
            # A[i,4], B[4,k], C[i,k]
            U[i * 5 + 4][col] = 1
            V[4 * 6 + k][col] = 1
            W[k * 4 + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
