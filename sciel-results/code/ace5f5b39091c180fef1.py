"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Strassen decomposition for 4x4 x 4x8
    # Rank 98 (2 * 49)
    n, m, p = DIMENSIONS
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 matrices
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For 4x4, we split into four 2x2 blocks.
    # Block indices:
    # A: (0,0), (0,1), (1,0), (1,1) -> 2x2 blocks
    # B: (0,0), (0,1), (1,0), (1,1) -> 2x2 blocks
    # C: (0,0), (0,1), (1,0), (1,1) -> 2x2 blocks
    
    # We perform two independent 4x4 multiplications for the 4x8 case.
    # Left half of B (cols 0-3) and Right half of B (cols 4-7).
    
    col_offset = 0
    for part in range(2):
        # Define the 7 Strassen terms for this part
        # Each term is a 2x2 matrix multiplication, which has rank 7.
        # So we need 7 * 7 = 49 columns for this part.
        
        # Strassen coefficients for A, B, C blocks
        # A blocks: A00, A01, A10, A11
        # B blocks: B00, B01, B10, B11
        # C blocks: C00, C01, C10, C11
        
        # M1: A00+A11, B00+B11 -> C00+=, C11+=
        # M2: A10+A11, B00 -> C10+=
        # M3: A00, B01-B11 -> C01+=
        # M4: A11, B10-B00 -> C10+=, C11-=
        # M5: A00+A01, B11 -> C00-=, C01+=
        # M6: A10-A00, B00+B01 -> C11+=
        # M7: A01-A11, B10+B11 -> C00+=, C11+=
        
        # We need to map these to the 2x2 rank-7 decomposition.
        # The 2x2 rank-7 decomposition is:
        # M1: (A00+A11)(B00+B11)
        # M2: (A10+A11)B00
        # M3: A00(B01-B11)
        # M4: A11(B10-B00)
        # M5: (A00+A01)B11
        # M6: (A10-A00)(B00+B01)
        # M7: (A01-A11)(B10+B11)
        
        # And the 2x2 rank-7 decomposition of the 2x2 multiplication itself is:
        # P1 = (A00+A11)(B00+B11)
        # P2 = (A10+A11)B00
        # P3 = A00(B01-B11)
        # P4 = A11(B10-B00)
        # P5 = (A00+A01)B11
        # P6 = (A10-A00)(B00+B01)
        # P7 = (A01-A11)(B10+B11)
        
        # Wait, the 7 terms ARE the Strassen terms.
        # So for each Strassen term (which is a 2x2 multiplication), we use the rank-7 decomposition.
        
        # Let's define the rank-7 decomposition of 2x2 multiplication.
        # U_2x2, V_2x2, W_2x2 of shape (4, 7), (4, 7), (4, 7).
        
        # U_2x2 rows: (0,0), (0,1), (1,0), (1,1)
        # V_2x2 rows: (0,0), (0,1), (1,0), (1,1)
        # W_2x2 rows: (0,0), (0,1), (1,0), (1,1)
        
        # P1: (A00+A11)(B00+B11)
        # U: A00=1, A11=1. V: B00=1, B11=1. W: C00=1, C11=1.
        # P2: (A10+A11)B00
        # U: A10=1, A11=1. V: B00=1. W: C10=1.
        # P3: A00(B01-B11)
        # U: A00=1. V: B01=1, B11=-1. W: C01=1.
        # P4: A11(B10-B00)
        # U: A11=1. V: B10=1, B00=-1. W: C10=1, C11=-1.
        # P5: (A00+A01)B11
        # U: A00=1, A01=1. V: B11=1. W: C00=-1, C01=1.
        # P6: (A10-A00)(B00+B01)
        # U: A10=1, A00=-1. V: B00=1, B01=1. W: C11=1.
        # P7: (A01-A11)(B10+B11)
        # U: A01=1, A11=-1. V: B10=1, B11=1. W: C00=1, C11=1.
        
        # Now we combine the two levels.
        # For each of the 7 Strassen terms (P1..P7), we have a 2x2 multiplication.
        # Each 2x2 multiplication is decomposed into 7 rank-1 terms.
        # So total 49 terms.
        
        # We need to map the indices.
        # A is 4x4. Blocks A00, A01, A10, A11 are 2x2.
        # A00: rows 0-1, cols 0-1.
        # A01: rows 0-1, cols 2-3.
        # A10: rows 2-3, cols 0-1.
        # A11: rows 2-3, cols 2-3.
        
        # B is 4x4 (for this part). Blocks B00, B01, B10, B11 are 2x2.
        # B00: rows 0-1, cols 0-1.
        # B01: rows 0-1, cols 2-3.
        # B10: rows 2-3, cols 0-1.
        # B11: rows 2-3, cols 2-3.
        
        # C is 4x4. Blocks C00, C01, C10, C11 are 2x2.
        # C00: rows 0-1, cols 0-1.
        # C01: rows 0-1, cols 2-3.
        # C10: rows 2-3, cols 0-1.
        # C11: rows 2-3, cols 2-3.
        
        # For each Strassen term s in 0..6:
        #   For each rank-1 term r in 0..6:
        #     col = col_offset + s*7 + r
        #     U, V, W are filled based on the combination.
        
        # Define the 2x2 rank-1 terms for A, B, C.
        # U_2x2[s, r, (i,j)]
        # V_2x2[s, r, (j,k)]
        # W_2x2[s, r, (i,k)]
        
        # We will hardcode the 49 combinations.
        
        # Helper to get 2x2 block indices
        def get_a_idx(block, i, j):
            # block 0: 00, 1: 01, 2: 10, 3: 11
            # i, j in 0..1
            if block == 0: return i * 2 + j
            if block == 1: return i * 2 + (j + 2)
            if block == 2: return (i + 2) * 2 + j
            if block == 3: return (i + 2) * 2 + (j + 2)
            
        def get_b_idx(block, i, j):
            # B is 4x4 in this part.
            # block 0: 00, 1: 01, 2: 10, 3: 11
            if block == 0: return i * 4 + j
            if block == 1: return i * 4 + (j + 2)
            if block == 2: return (i + 2) * 4 + j
            if block == 3: return (i + 2) * 4 + (j + 2)
            
        def get_c_idx(block, i, j):
            # C is 4x4 in this part.
            # W index is k*n + i. Here n=4.
            # k is col, i is row.
            # block 0: 00, 1: 01, 2: 10, 3: 11
            if block == 0: return j * 4 + i
            if block == 1: return (j + 2) * 4 + i
            if block == 2: return j * 4 + (i + 2)
            if block == 3: return (j + 2) * 4 + (i + 2)
            
        # Define the 7 Strassen terms for the 4x4 level
        # Each term specifies which 2x2 blocks of A, B, C are involved and with what coefficients.
        # A_blocks: list of (block_idx, coeff)
        # B_blocks: list of (block_idx, coeff)
        # C_blocks: list of (block_idx, coeff)
        
        strassen_terms = [
            # M1
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            # M2
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1)]),
            # M3
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1)]),
            # M4
            ([(3, 1)], [(2, 1), (0, -1)], [(2, 1), (3, -1)]),
            # M5
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            # M6
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            # M7
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (3, 1)])
        ]
        
        # Define the 7 rank-1 terms for the 2x2 level
        # Each term specifies which 2x2 elements of A, B, C are involved.
        # A_els: list of (i, j, coeff)
        # B_els: list of (i, j, coeff)
        # C_els: list of (i, j, coeff)
        
        rank1_terms = [
            # P1
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            # P2
            ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)]),
            # P3
            ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1)]),
            # P4
            ([(1,1,1)], [(1,0,1), (0,0,-1)], [(1,0,1), (1,1,-1)]),
            # P5
            ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
            # P6
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
            # P7
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1), (1,1,1)])
        ]
        
        for s in range(7):
            a_blocks, b_blocks, c_blocks = strassen_terms[s]
            for r in range(7):
                col = col_offset + s * 7 + r
                a_els, b_els, c_els = rank1_terms[r]
                
                # Fill U
                for a_block_idx, a_block_coeff in a_blocks:
                    for ai, aj, a_el_coeff in a_els:
                        idx = get_a_idx(a_block_idx, ai, aj)
                        U[idx][col] += a_block_coeff * a_el_coeff
                        
                # Fill V
                for b_block_idx, b_block_coeff in b_blocks:
                    for bi, bj, b_el_coeff in b_els:
                        idx = get_b_idx(b_block_idx, bi, bj)
                        V[idx][col] += b_block_coeff * b_el_coeff
                        
                # Fill W
                for c_block_idx, c_block_coeff in c_blocks:
                    for ci, cj, c_el_coeff in c_els:
                        idx = get_c_idx(c_block_idx, ci, cj)
                        W[idx][col] += c_block_coeff * c_el_coeff
        
        col_offset += 49
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
