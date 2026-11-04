"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # We will use a hybrid approach:
    # Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1).
    # Compute C1 = A*B1 using Strassen 4x4x4 (rank 28).
    # Compute C2 = A*B2 using Strassen 4x4x4 (rank 28).
    # Compute C3 = A*B3 using Schoolbook 4x4x1 (rank 16).
    # Total rank = 28 + 28 + 16 = 72.
    
    rank = 72
    U = [[0] * rank for _ in range(n * m)]  # 16 x 72
    V = [[0] * rank for _ in range(m * p)]  # 36 x 72
    W = [[0] * rank for _ in range(p * n)]  # 36 x 72
    
    col_idx = 0
    
    # Helper to add a 2x2x2 schoolbook decomposition
    # a_off: offset in U for 2x2 A block (4 rows: 0,1,4,5 relative to a_off)
    # b_off: offset in V for 2x2 B block (4 rows: 0,1,4,5 relative to b_off)
    # c_off: offset in W for 2x2 C block (4 rows: 0,1,4,5 relative to c_off)
    # Returns new col_idx
    def add_2x2_schoolbook(a_off, b_off, c_off, c_start):
        c = c_start
        # C00 = A00 B00 + A01 B10
        # C01 = A00 B01 + A01 B11
        # C10 = A10 B00 + A11 B10
        # C11 = A10 B01 + A11 B11
        
        # Product 1: A00 B00 -> C00, C10? No.
        # Schoolbook for 2x2:
        # P1 = A00 B00
        # P2 = A00 B01
        # P3 = A01 B00
        # P4 = A01 B01
        # P5 = A10 B00
        # P6 = A10 B01
        # P7 = A11 B00
        # P8 = A11 B01
        
        # C00 = P1 + P3
        # C01 = P2 + P4
        # C10 = P5 + P7
        # C11 = P6 + P8
        
        # U indices for 2x2 A block starting at a_off:
        # A00: a_off+0
        # A01: a_off+1
        # A10: a_off+4
        # A11: a_off+5
        
        # V indices for 2x2 B block starting at b_off:
        # B00: b_off+0
        # B01: b_off+1
        # B10: b_off+4
        # B11: b_off+5
        
        # W indices for 2x2 C block starting at c_off:
        # C00: c_off+0
        # C01: c_off+1
        # C10: c_off+4
        # C11: c_off+5
        
        # P1: A00 * B00
        U[a_off+0][c] = 1
        V[b_off+0][c] = 1
        W[c_off+0][c] = 1 # C00
        W[c_off+4][c] = 1 # C10
        c += 1
        
        # P2: A00 * B01
        U[a_off+0][c] = 1
        V[b_off+1][c] = 1
        W[c_off+1][c] = 1 # C01
        W[c_off+5][c] = 1 # C11
        c += 1
        
        # P3: A01 * B00
        U[a_off+1][c] = 1
        V[b_off+0][c] = 1
        W[c_off+0][c] = 1 # C00
        W[c_off+4][c] = 1 # C10
        c += 1
        
        # P4: A01 * B01
        U[a_off+1][c] = 1
        V[b_off+1][c] = 1
        W[c_off+1][c] = 1 # C01
        W[c_off+5][c] = 1 # C11
        c += 1
        
        # P5: A10 * B00
        U[a_off+4][c] = 1
        V[b_off+0][c] = 1
        W[c_off+4][c] = 1 # C10
        W[c_off+0][c] = 1 # C00? No, C10 = A10 B00 + A11 B10.
        # Wait, my mapping above:
        # C00 = A00 B00 + A01 B10.
        # My P1, P3 used B00.
        # Let's stick to standard schoolbook:
        # C[i][j] = sum_k A[i][k] B[k][j]
        # C00 = A00 B00 + A01 B10
        # C01 = A00 B01 + A01 B11
        # C10 = A10 B00 + A11 B10
        # C11 = A10 B01 + A11 B11
        
        # Let's restart the 2x2 schoolbook helper with correct terms.
        # The previous implementation was mixing up indices.
        
        # We will define the 8 products explicitly.
        
        # Reset c to c_start for the helper logic, but we are inside the function.
        # I will rewrite the helper properly in the main code block.
        return c

    # Correct 2x2 Schoolbook Helper
    def add_2x2_schoolbook_correct(a_off, b_off, c_off, c_start):
        c = c_start
        # A00: a_off+0, A01: a_off+1, A10: a_off+4, A11: a_off+5
        # B00: b_off+0, B01: b_off+1, B10: b_off+4, B11: b_off+5
        # C00: c_off+0, C01: c_off+1, C10: c_off+4, C11: c_off+5
        
        # 1. A00 B00 -> C00, C10? No. C00 has A00 B00. C10 has A10 B00.
        # The products are:
        # P1 = A00 B00
        # P2 = A00 B10
        # P3 = A01 B00
        # P4 = A01 B10
        # P5 = A00 B01
        # P6 = A00 B11
        # P7 = A01 B01
        # P8 = A01 B11
        # P9 = A10 B00
        # P10 = A10 B10
        # P11 = A11 B00
        # P12 = A11 B10
        # P13 = A10 B01
        # P14 = A10 B11
        # P15 = A11 B01
        # P16 = A11 B11
        
        # This is 16 products. Rank 16.
        # We can do better? No, 2x2x2 schoolbook is rank 4? No, 2x2x2 schoolbook is 4 products?
        # C00 = A00 B00 + A01 B10 (2 terms)
        # C01 = A00 B01 + A01 B11 (2 terms)
        # C10 = A10 B00 + A11 B10 (2 terms)
        # C11 = A10 B01 + A11 B11 (2 terms)
        # Total 8 terms? No, 4 sums of 2 products.
        # We can compute 4 products?
        # M1 = A00 B00
        # M2 = A01 B10
        # M3 = A00 B01
        # M4 = A01 B11
        # M5 = A10 B00
        # M6 = A11 B10
        # M7 = A10 B01
        # M8 = A11 B11
        # Yes, 8 scalar multiplications.
        
        # Let's implement 8 scalar multiplications.
        
        # 1. A00 B00
        U[a_off+0][c] = 1
        V[b_off+0][c] = 1
        W[c_off+0][c] = 1 # C00
        c += 1
        
        # 2. A01 B10
        U[a_off+1][c] = 1
        V[b_off+4][c] = 1
        W[c_off+0][c] = 1 # C00
        c += 1
        
        # 3. A00 B01
        U[a_off+0][c] = 1
        V[b_off+1][c] = 1
        W[c_off+1][c] = 1 # C01
        c += 1
        
        # 4. A01 B11
        U[a_off+1][c] = 1
        V[b_off+5][c] = 1
        W[c_off+1][c] = 1 # C01
        c += 1
        
        # 5. A10 B00
        U[a_off+4][c] = 1
        V[b_off+0][c] = 1
        W[c_off+4][c] = 1 # C10
        c += 1
        
        # 6. A11 B10
        U[a_off+5][c] = 1
        V[b_off+4][c] = 1
        W[c_off+4][c] = 1 # C10
        c += 1
        
        # 7. A10 B01
        U[a_off+4][c] = 1
        V[b_off+1][c] = 1
        W[c_off+5][c] = 1 # C11
        c += 1
        
        # 8. A11 B11
        U[a_off+5][c] = 1
        V[b_off+5][c] = 1
        W[c_off+5][c] = 1 # C11
        c += 1
        
        return c

    # Helper to add 4x4x4 Strassen decomposition
    def add_4x4_strassen(b_col_offset, c_col_offset, c_start):
        # A is always 0..15 in U.
        # B block is at b_col_offset in V. (4x4 block, 16 rows)
        # C block is at c_col_offset in W. (4x4 block, 16 rows)
        
        c = c_start
        
        # Block offsets in V (4x4 matrix B stored in 16 rows)
        # B11: 0,1,4,5
        # B12: 2,3,6,7
        # B21: 8,9,12,13
        # B22: 10,11,14,15
        
        # Block offsets in W (4x4 matrix C stored in 16 rows)
        # C11: 0,1,4,5
        # C12: 2,3,6,7
        # C21: 8,9,12,13
        # C22: 10,11,14,15
        
        # A blocks in U
        # A11: 0,1,4,5
        # A12: 2,3,6,7
        # A21: 8,9,12,13
        # A22: 10,11,14,15
        
        # We need to form linear combinations for Strassen.
        # M1 = (A11+A22)(B11+B22)
        # A11+A22:
        #   (0,0): A00+A20 (U[0]+U[8])
        #   (0,1): A01+A21 (U[1]+U[9])
        #   (1,0): A10+A30 (U[4]+U[12])
        #   (1,1): A11+A31 (U[5]+U[13])
        # B11+B22:
        #   (0,0): B00+B20 (V[b+0]+V[b+8])
        #   (0,1): B01+B21 (V[b+1]+V[b+9])
        #   (1,0): B10+B30 (V[b+4]+V[b+12])
        #   (1,1): B11+B31 (V[b+5]+V[b+13])
        
        # We can't just add offsets. We need to create the linear combos in U and V.
        # This means we need to assign specific columns to these products.
        
        # Strategy: For each of the 7 Strassen products, we create 4 scalar multiplications
        # corresponding to the 4 entries of the 2x2 result.
        # However, the inputs are linear combinations.
        # So for M1, we have:
        # M1_00 = (A00+A20)(B00+B20) + (A01+A21)(B10+B30) ? No.
        # M1 is a 2x2 matrix.
        # M1[0,0] = (A11+A22)[0,0] * (B11+B22)[0,0] + (A11+A22)[0,1] * (B11+B22)[1,0]
        # This is getting complicated to map directly to scalar products without expanding.
        
        # Alternative: Expand the Strassen formula into scalar products.
        # M1 = (A11+A22)(B11+B22)
        # M1 = A11 B11 + A11 B22 + A22 B11 + A22 B22
        # Each term is a 2x2 matrix product.
        # A11 B11: 4 scalar mults.
        # A11 B22: 4 scalar mults.
        # A22 B11: 4 scalar mults.
        # A22 B22: 4 scalar mults.
        # Total 16 scalar mults for M1?
        # No, Strassen reduces 8 matrix mults to 7.
        # But here we are at scalar level.
        # Standard Strassen for 4x4x4 (using 2x2 blocks) has rank 28.
        # This implies we can do it with 28 scalar multiplications.
        # This requires careful construction of the U, V, W matrices.
        
        # Let's use a known rank-28 decomposition structure.
        # Or, simpler: just use schoolbook for 4x4x4? Rank 16.
        # 16 + 16 + 16 = 48.
        # This is still better than 72? No, 48 < 72.
        # Wait, schoolbook for 4x4x4 is 16.
        # If I use schoolbook for the two 4x4 blocks and schoolbook for the 4x1 block:
        # 16 + 16 + 16 = 48.
        # This is much simpler and safer than Strassen!
        
        return c

    # Let's stick to Schoolbook for everything to ensure correctness.
    # Rank 48 is a huge improvement over 144.
    
    # Recalculate rank for all schoolbook:
    # Block 1: 4x4 x 4x4 -> 16
    # Block 2: 4x4 x 4x4 -> 16
    # Block 3: 4x4 x 4x1 -> 16
    # Total 48.
    
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Block 1: B cols 0-3, C cols 0-3
    # A is 0-15
    # B is 0-15 (in V, offset 0)
    # C is 0-15 (in W, offset 0)
    # We need to map 4x4 schoolbook.
    # C[i,j] = sum_k A[i,k] B[k,j]
    # i in 0..3, j in 0..3, k in 0..3
    
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*4 + k][col_idx] = 1
                V[k*4 + j][col_idx] = 1
                W[j*4 + i][col_idx] = 1
                col_idx += 1
                
    # Block 2: B cols 4-7, C cols 4-7
    # A is 0-15
    # B is 16-31 (in V, offset 16)
    # C is 16-31 (in W, offset 16)
    
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*4 + k][col_idx] = 1
                V[16 + k*4 + j][col_idx] = 1
                W[16 + j*4 + i][col_idx] = 1
                col_idx += 1
                
    # Block 3: B col 8, C col 8
    # A is 0-15
    # B is 32-35 (in V, offset 32)
    # C is 32-35 (in W, offset 32)
    
    for i in range(4):
        for k in range(4):
            U[i*4 + k][col_idx] = 1
            V[32 + k][col_idx] = 1
            W[32 + i][col_idx] = 1
            col_idx += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
