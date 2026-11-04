DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Two independent 4x4x4 multiplications
    # Each 4x4x4 uses 8 Strassen 2x2x2 blocks, rank 8*7=56
    # Total rank 56 + 56 = 112
    rank = 112
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map this to the 4x4x4 case.
    # A 4x4 matrix is split into four 2x2 blocks.
    # A = [[A00, A01], [A10, A11]] where Aij are 2x2.
    # B = [[B00, B01], [B10, B11]] where Bij are 2x2.
    # C = [[C00, C01], [C10, C11]] where Cij are 2x2.
    
    # We have 8 such block multiplications.
    # For each block multiplication, we use 7 Strassen products.
    # Total 56 products for one 4x4x4.
    
    # Let's define the 7 Strassen vectors for a 2x2x2 block.
    # U_strassen: 4x7, V_strassen: 4x7, W_strassen: 4x7
    # Indices 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1)
    
    U_s = [
        [1, 0, 1, 0, 1, 0, 0], # A00
        [0, 0, 0, 0, 1, 0, 1], # A01
        [0, 1, 0, 0, 0, 1, 0], # A10
        [1, 1, 0, 1, 0, 0, -1] # A11
    ]
    V_s = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, -1, 1, 0, 1, 0], # B01
        [0, 0, 1, 0, 0, 0, 1], # B10
        [1, 0, -1, 0, 1, 0, 1] # B11
    ]
    W_s = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0], # C01
        [0, 1, 0, 1, 0, 0, 0], # C10
        [1, -1, 1, 0, 0, 1, 0] # C11
    ]
    
    # Helper to add a 2x2 block decomposition at offset
    def add_block(base_U, base_V, base_W, i_off, j_off, k_off, rank_off):
        # i_off, j_off, k_off are the top-left indices of the 2x2 block in the 4x4 matrix
        # rank_off is the starting column in U, V, W
        for di in range(2):
            for dj in range(2):
                u_idx = (i_off + di) * 4 + (j_off + dj)
                for r in range(7):
                    base_U[u_idx][rank_off + r] += U_s[di*2+dj][r]
        for dj in range(2):
            for dk in range(2):
                v_idx = (j_off + dj) * 8 + (k_off + dk)
                for r in range(7):
                    base_V[v_idx][rank_off + r] += V_s[dj*2+dk][r]
        for dk in range(2):
            for di in range(2):
                w_idx = (k_off + dk) * 4 + (i_off + di)
                for r in range(7):
                    base_W[w_idx][rank_off + r] += W_s[dk*2+di][r]

    # The 4x4x4 multiplication C = A*B
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # We need to decompose each term? No, Strassen decomposes the whole 2x2 block product.
    # But C00 is a sum of two products.
    # We can't just use Strassen on A00*B00 and A01*B10 separately and add them?
    # Yes, we can. The rank is additive.
    # So for C00, we use 7 products for A00*B00 and 7 products for A01*B10.
    # Total 14 products for C00.
    # Similarly for C01, C10, C11.
    # Total 4 * 14 = 56 products for 4x4x4.
    
    # Block 0: A00*B00 -> C00 (part 1)
    add_block(U, V, W, 0, 0, 0, 0)
    # Block 1: A01*B10 -> C00 (part 2)
    # A01 is at (0, 2), B10 is at (2, 0)
    # Wait, A is 4x4. A00 is rows 0-1, cols 0-1. A01 is rows 0-1, cols 2-3.
    # B is 4x8. For the first 4x4x4, B is 4x4 (cols 0-3).
    # B00 is rows 0-1, cols 0-1. B10 is rows 2-3, cols 0-1.
    # So A01*B10: A rows 0-1, cols 2-3. B rows 2-3, cols 0-1.
    # Output C00: rows 0-1, cols 0-1.
    add_block(U, V, W, 0, 2, 0, 7)
    
    # Block 2: A00*B01 -> C01 (part 1)
    # A00: (0,0). B01: rows 0-1, cols 2-3.
    # Output C01: rows 0-1, cols 2-3.
    add_block(U, V, W, 0, 0, 2, 14)
    
    # Block 3: A01*B11 -> C01 (part 2)
    # A01: (0,2). B11: rows 2-3, cols 2-3.
    # Output C01: rows 0-1, cols 2-3.
    add_block(U, V, W, 0, 2, 2, 21)
    
    # Block 4: A10*B00 -> C10 (part 1)
    # A10: rows 2-3, cols 0-1. B00: rows 0-1, cols 0-1.
    # Output C10: rows 2-3, cols 0-1.
    add_block(U, V, W, 2, 0, 0, 28)
    
    # Block 5: A11*B10 -> C10 (part 2)
    # A11: rows 2-3, cols 2-3. B10: rows 2-3, cols 0-1.
    # Output C10: rows 2-3, cols 0-1.
    add_block(U, V, W, 2, 2, 0, 35)
    
    # Block 6: A10*B01 -> C11 (part 1)
    # A10: (2,0). B01: (0,2).
    # Output C11: rows 2-3, cols 2-3.
    add_block(U, V, W, 2, 0, 2, 42)
    
    # Block 7: A11*B11 -> C11 (part 2)
    # A11: (2,2). B11: (2,2).
    # Output C11: rows 2-3, cols 2-3.
    add_block(U, V, W, 2, 2, 2, 49)
    
    # This is the first 4x4x4 (cols 0-3 of B).
    # Now for the second 4x4x4 (cols 4-7 of B).
    # Same A, different B.
    # B' is 4x4 (cols 4-7).
    # C' is 4x4 (cols 4-7).
    
    # Block 8: A00*B'00 -> C'00
    add_block(U, V, W, 0, 0, 4, 56)
    # Block 9: A01*B'10 -> C'00
    add_block(U, V, W, 0, 2, 4, 63)
    # Block 10: A00*B'01 -> C'01
    add_block(U, V, W, 0, 0, 6, 70)
    # Block 11: A01*B'11 -> C'01
    add_block(U, V, W, 0, 2, 6, 77)
    # Block 12: A10*B'00 -> C'10
    add_block(U, V, W, 2, 0, 4, 84)
    # Block 13: A11*B'10 -> C'10
    add_block(U, V, W, 2, 2, 4, 91)
    # Block 14: A10*B'01 -> C'11
    add_block(U, V, W, 2, 0, 6, 98)
    # Block 15: A11*B'11 -> C'11
    add_block(U, V, W, 2, 2, 6, 105)
    
    return {"U": U, "V": V, "W": W}
