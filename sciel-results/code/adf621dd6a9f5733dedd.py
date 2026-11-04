"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Strategy: Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
    # Use Strassen-based rank-28 decomposition for 4x4x4 twice
    # Use schoolbook rank-16 for 4x4x1
    # Total rank: 28 + 28 + 16 = 72
    
    rank = 72
    U = [[0] * rank for _ in range(n * m)]  # 16 x 72
    V = [[0] * rank for _ in range(m * p)]  # 36 x 72
    W = [[0] * rank for _ in range(p * n)]  # 36 x 72
    
    col = 0
    
    def add_strassen_2x2(U, V, W, a_row_offset, b_row_offset, c_row_offset, col_start):
        """
        Add Strassen 2x2x2 decomposition (rank 7).
        a_row_offset: row offset in U for the 2x2 A block (4 rows: a00,a01,a10,a11)
        b_row_offset: row offset in V for the 2x2 B block (4 rows: b00,b01,b10,b11)
        c_row_offset: row offset in W for the 2x2 C block (4 rows: c00,c01,c10,c11)
        col_start: starting column index
        Returns: new column index
        """
        c = col_start
        # M1 = (A11+A22)(B11+B22)
        U[a_row_offset + 0][c] = 1
        U[a_row_offset + 3][c] = 1
        V[b_row_offset + 0][c] = 1
        V[b_row_offset + 3][c] = 1
        W[c_row_offset + 0][c] = 1   # C11
        W[c_row_offset + 3][c] = 1   # C22
        c += 1
        
        # M2 = (A21+A22)B11
        U[a_row_offset + 2][c] = 1
        U[a_row_offset + 3][c] = 1
        V[b_row_offset + 0][c] = 1
        W[c_row_offset + 2][c] = 1   # C21
        W[c_row_offset + 3][c] = -1  # C22
        c += 1
        
        # M3 = A11(B21-B22)
        U[a_row_offset + 0][c] = 1
        V[b_row_offset + 2][c] = 1
        V[b_row_offset + 3][c] = -1
        W[c_row_offset + 1][c] = 1   # C12
        W[c_row_offset + 3][c] = 1   # C22
        c += 1
        
        # M4 = A22(B12-B11)
        U[a_row_offset + 3][c] = 1
        V[b_row_offset + 1][c] = 1
        V[b_row_offset + 0][c] = -1
        W[c_row_offset + 0][c] = 1   # C11
        W[c_row_offset + 2][c] = 1   # C21
        c += 1
        
        # M5 = (A11+A12)B22
        U[a_row_offset + 0][c] = 1
        U[a_row_offset + 1][c] = 1
        V[b_row_offset + 3][c] = 1
        W[c_row_offset + 0][c] = -1  # C11
        W[c_row_offset + 1][c] = 1   # C12
        c += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[a_row_offset + 2][c] = 1
        U[a_row_offset + 0][c] = -1
        V[b_row_offset + 0][c] = 1
        V[b_row_offset + 1][c] = 1
        W[c_row_offset + 3][c] = 1   # C22
        c += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[a_row_offset + 1][c] = 1
        U[a_row_offset + 3][c] = -1
        V[b_row_offset + 2][c] = 1
        V[b_row_offset + 3][c] = 1
        W[c_row_offset + 0][c] = 1   # C11
        c += 1
        
        return c
    
    def add_4x4x4_strassen(U, V, W, a_offset, b_offset, c_offset, col_start):
        """
        Add 28-rank Strassen-based decomposition for 4x4x4.
        Uses Strassen for the 2x2 block matrix multiplication,
        with schoolbook (rank 4) for each inner 2x2x2 multiplication.
        
        a_offset: row offset in U for the 4x4 A block (16 rows)
        b_offset: row offset in V for the 4x4 B block (16 rows)
        c_offset: row offset in W for the 4x4 C block (16 rows)
        col_start: starting column
        Returns: new column
        """
        c = col_start
        
        # For 4x4 matrix A stored with row index = i*4 + j:
        # A11 = A[0:2, 0:2] -> U[a_offset+0], U[a_offset+1], U[a_offset+4], U[a_offset+5]
        # A12 = A[0:2, 2:4] -> U[a_offset+2], U[a_offset+3], U[a_offset+6], U[a_offset+7]
        # A21 = A[2:4, 0:2] -> U[a_offset+8], U[a_offset+9], U[a_offset+12], U[a_offset+13]
        # A22 = A[2:4, 2:4] -> U[a_offset+10], U[a_offset+11], U[a_offset+14], U[a_offset+15]
        
        # Similarly for B and C.
        
        # We need to compute 7 Strassen products, each a 2x2 matrix product.
        # For each product, we use schoolbook (4 scalar multiplications).
        
        # Define helper to get the 4 U-indices for a 2x2 subblock of A
        def get_A_block(r0, r1, c0, c1):
            """Return list of 4 U row indices for A[r0:r1, c0:c1]"""
            return [a_offset + r0*4 + c0, a_offset + r0*4 + c1, 
                    a_offset + r1*4 + c0, a_offset + r1*4 + c1]
        
        def get_B_block(r0, r1, c0, c1):
            """Return list of 4 V row indices for B[r0:r1, c0:c1]"""
            return [b_offset + r0*4 + c0, b_offset + r0*4 + c1, 
                    b_offset + r1*4 + c0, b_offset + r1*4 + c1]
        
        def get_C_block(r0, r1, c0, c1):
            """Return list of 4 W row indices for C[r0:r1, c0:c1]"""
            return [c_offset + r0*4 + c0, c_offset + r0*4 + c1, 
                    c_offset + r1*4 + c0, c_offset + r1*4 + c1]
        
        # A11: rows 0-1, cols 0-1
        A11 = get_A_block(0, 1, 0, 1)  # [a00, a01, a10, a11]
        # A12: rows 0-1, cols 2-3
        A12 = get_A_block(0, 1, 2, 3)  # [a02, a03, a12, a13]
        # A21: rows 2-3, cols 0-1
        A21 = get_A_block(2, 3, 0, 1)  # [a20, a21, a30, a31]
        # A22: rows 2-3, cols 2-3
        A22 = get_A_block(2, 3, 2, 3)  # [a22, a23, a32, a33]
        
        # B11: rows 0-1, cols 0-1
        B11 = get_B_block(0, 1, 0, 1)
        # B12: rows 0-1, cols 2-3
        B12 = get_B_block(0, 1, 2, 3)
        # B21: rows 2-3, cols 0-1
        B21 = get_B_block(2, 3, 0, 1)
        # B22: rows 2-3, cols 2-3
        B22 = get_B_block(2, 3, 2, 3)
        
        # C11: rows 0-1, cols 0-1
        C11 = get_C_block(0, 1, 0, 1)
        # C12: rows 0-1, cols 2-3
        C12 = get_C_block(0, 1, 2, 3)
        # C21: rows 2-3, cols 0-1
        C21 = get_C_block(2, 3, 0, 1)
        # C22: rows 2-3, cols 2-3
        C22 = get_C_block(2, 3, 2, 3)
        
        # Strassen formulas for 2x2 block matrix multiplication:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For each Mi, we compute a 2x2 matrix product using schoolbook (4 scalar mults).
        # We need to create virtual 2x2 matrices for each term.
        
        # M1 = (A11+A22)(B11+B22)
        # A11+A22 has entries:
        #   (A11+A22)[0,0] = A11[0,0] + A22[0,0] = A11[0] + A22[0]
        #   (A11+A22)[0,1] = A11[0,1] + A22[0,1] = A11[1] + A22[1]
        #   (A11+A22)[1,0] = A11[1,0] + A22[1,0] = A11[2] + A22[2]
        #   (A11+A22)[1,1] = A11[1,1] + A22[1,1] = A11[3] + A22[3]
        # B11+B22 has entries:
        #   (B11+B22)[0,0] = B11[0] + B22[0]
        #   (B11+B22)[0,1] = B11[1] + B22[1]
        #   (B11+B22)[1,0] = B11[2] + B22[2]
        #   (B11+B22)[1,1] = B11[3] + B22[3]
        
        # Schoolbook for (A11+A22)(B11+B22):
        # Result[0,0] = (A11+A22)[0,0]*(B11+B22)[0,0] + (A11+A22)[0,1]*(B11+B22)[1,0]
        # This gets complex. Let me use a different approach.
        
        # Instead of expanding each term, I'll use the fact that we can compute
        # each 2x2 product with 4 scalar multiplications by treating the linear
        # combinations as part of the U and V matrices.
        
        # Actually, the cleanest approach: for each of the 7 Strassen products,
        # create 4 new scalar multiplication terms.
        
        # Let me define a helper that adds a 2x2 schoolbook multiplication
        # where the A and B blocks are given as lists of (U_index, coefficient) pairs.
        
        def add_schoolbook_2x2(a_terms, b_terms, c_target, col_start):
            """
            Add 4 scalar multiplications for a 2x2 matrix product.
            a_terms: list of 4 (U_row_index, coeff) for A[0,0], A[0,1], A[1,0], A[1,1]
            b_terms: list of 4 (V_row_index, coeff) for B[0,0], B[0,1], B[1,0], B[1,1]
            c_target: list of 4 W row indices for C[0,0], C[0,1], C[1,0], C[1,1]
            col_start: starting column
            Returns: new column
            """
            c = col_start
            # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
            # We can't do two multiplications in one rank-1 term.
            # So we need 4 separate rank-1 terms for schoolbook 2x2x2.
            
            # Term 1: A[0,0] * B[0,0] -> contributes to C[0,0]
            for (a_idx, a_coeff), (b_idx, b_coeff) in [(a_terms[0], b_terms[0])]:
                if a_coeff != 0 and b_coeff != 0:
                    U[a_idx][c] = a_coeff
                    V[b_idx][c] = b_coeff
                    W[c_target[0]][c] = 1
            c += 1
            
            # Term 2: A[0,1] * B[1,0] -> contributes to C[0,0]
            for (a_idx, a_coeff), (b_idx, b_coeff) in [(a_terms[1], b_terms[2])]:
                if a_coeff != 0 and b_coeff != 0:
                    U[a_idx][c] = a_coeff
                    V[b_idx][c] = b_coeff
                    W[c_target[0]][c] = 1
            c += 1
            
            # Term 3: A[1,0] * B[0,0] -> contributes to C[1,0]
            for (a_idx, a_coeff), (b_idx, b_coeff) in [(a_terms[2], b_terms[0])]:
                if a_coeff != 0 and b_coeff != 0:
                    U[a_idx][c] = a_coeff
                    V[b_idx][c] = b_coeff
                    W[c_target[2]][c] = 1
            c += 1
            
            # Term 4: A[1,1] * B[1,0] -> contributes to C[1,0]
            for (a_idx, a_coeff), (b_idx, b_coeff) in [(a_terms[3], b_terms[2])]:
                if a_coeff != 0 and b_coeff != 0:
                    U[a_idx][c] = a_coeff
                    V[b_idx][c] = b_coeff
                    W[c_target[2]][c] = 1
            c += 1
            
            # Term 5: A[0,0] * B[0,1] -> contributes to C[0,1]
            for (a_idx, a_coeff), (b_idx, b_coeff) in [(a_terms[0], b_terms[1])]:
                if a_coeff != 0 and b_coeff != 0:
                    U[a_idx][c] = a_coeff
                    V[b_idx][c] = b_coeff
                    W[c_target[1]][c] = 1
            c += 1
            
            # Term 6: A[0,1] * B[1,1] -> contributes to C[0,1]
            for (a_idx, a_coeff), (b_idx, b_coeff) in [(a_terms[1], b_terms[3])]:
                if a_coeff != 0 and b_coeff != 0:
                    U[a_idx][c] = a_coeff
                    V[b_idx][c] = b_coeff
                    W[c_target[1]][c] = 1
            c += 1
            
            # Term 7: A[1,0] * B[0,1] -> contributes to C[1,1]
            for (a_idx, a_coeff), (b_idx, b_coeff) in [(a_terms[2], b_terms[1])]:
                if a_coeff != 0 and b_coeff != 0:
                    U[a_idx][c] = a_coeff
                    V[b_idx][c] = b_coeff
                    W[c_target[3]][c] = 1
            c += 1
            
            # Term 8: A[1,1] * B[1,1] -> contributes to C[1,1]
            for (a_idx, a_coeff), (b_idx, b_coeff) in [(a_terms[3], b_terms[3])]:
                if a_coeff != 0 and b_coeff != 0:
                    U[a_idx][c] = a_coeff
                    V[b_idx][c] = b_coeff
                    W[c_target[3]][c] = 1
            c += 1
            
            return c
        
        # This approach uses 8 multiplications per 2x2 product, not 4.
        # I need to use Strassen for the inner 2x2 products too, or find a better way.
        
        # Actually, for schoolbook 2x2x2, we need exactly 4 scalar multiplications:
        # M1 = A[0,0]*B[0,0]
        # M2 = A[0,1]*B[1,0]  
        # M3 = A[1,0]*B[0,0]
        # M4 = A[1,1]*B[1,0]
        # But wait, that's not right. Schoolbook 2x2x2 needs 4 multiplications:
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
        
        # That's 8 scalar multiplications! Not 4.
        # I was wrong. Schoolbook 2x2x2 has rank 8, not 4.
        
        # Let me reconsider. The rank of 2x2x2 matrix multiplication is 7 (Strassen).
        # Schoolbook is 8.
        
        # So for 4x4x4 using Strassen for outer (7 products) and Strassen for inner (7 each):
        # Total rank = 7 * 7 = 49
        
        # Or Strassen for outer and schoolbook for inner: 7 * 8 = 56
        
        # Let me use Strassen for both: rank 49 for 4x4x4.
        # For 4x4x9: 49 + 49 + 16 = 114 (schoolbook for 4x4x1 is 16)
        
        # That's better than 144!
        
        # Let me implement rank 49 for 4x4x4.
        
        # Actually, let me just use a simpler approach: 
        # Use Strassen directly on the 4x4x4 by viewing it as 2x2 blocks,
        # and use Strassen (rank 7) for each inner 2x2x2.
        
        # Total: 7 * 7 = 49
        
        # For the 4x4x1 part, use schoolbook: 4*4*1 = 16
        
        # Total rank: 49 + 49 + 16 = 114
        
        # Let me rewrite with this approach.
        
        return c
    
    # Given the complexity, let me use a cleaner approach.
    # I'll implement the full Strassen-based decomposition properly.
    
    # Reset and start over with a cleaner implementation
    rank = 114  # 49 + 49 + 16
    U = [[0] * rank for _ in range(n * m)]  # 16 x 114
    V = [[0] * rank for _ in range(m * p)]  # 36 x 114
    W = [[0] * rank for _ in range(p * n)]  # 36 x 114
    
    col = 0
    
    # Helper: add Strassen 2x2x2 decomposition (rank 7)
    def add_strassen_2x2(U, V, W, a_rows, b_rows, c_rows, col_start):
        """
        Add Strassen 2x2x2 decomposition (rank 7).
        a_rows: list of 4 U row indices [a00, a01, a10, a11]
        b_rows: list of 4 V row indices [b00, b01, b10, b11]
        c_rows: list of 4 W row indices [c00, c01, c10, c11]
        col_start: starting column index
        Returns: new column index
        """
        c = col_start
        a00, a01, a10, a11 = a_rows
        b00, b01, b10, b11 = b_rows
        c00, c01, c10, c11 = c_rows
        
        # M1 = (A11+A22)(B11+B22)
        U[a00][c] = 1
        U[a11][c] = 1
        V[b00][c] = 1
        V[b11][c] = 1
        W[c00][c] = 1   # C11
        W[c11][c] = 1   # C22
        c += 1
        
        # M2 = (A21+A22)B11
        U[a10][c] = 1
        U[a11][c] = 1
        V[b00][c] = 1
        W[c10][c] = 1   # C21
        W[c11][c] = -1  # C22
        c += 1
        
        # M3 = A11(B21-B22)
        U[a00][c] = 1
        V[b10][c] = 1
        V[b11][c] = -1
        W[c01][c] = 1   # C12
        W[c11][c] = 1   # C22
        c += 1
        
        # M4 = A22(B12-B11)
        U[a11][c] = 1
        V[b01][c] = 1
        V[b00][c] = -1
        W[c00][c] = 1   # C11
        W[c10][c] = 1   # C21
        c += 1
        
        # M5 = (A11+A12)B22
        U[a00][c] = 1
        U[a01][c] = 1
        V[b11][c] = 1
        W[c00][c] = -1  # C11
        W[c01][c] = 1   # C12
        c += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[a10][c] = 1
        U[a00][c] = -1
        V[b00][c] = 1
        V[b01][c] = 1
        W[c11][c] = 1   # C22
        c += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[a01][c] = 1
        U[a11][c] = -1
        V[b10][c] = 1
        V[b11][c] = 1
        W[c00][c] = 1   # C11
        c += 1
        
        return c
    
    # Helper: get 4 row indices for a 2x2 subblock
    def get_block_rows(base_offset, r0, r1, c0, c1, stride):
        """
        Get 4 row indices for a 2x2 subblock.
        base_offset: base row offset
        r0, r1: row range
        c0, c1: col range
        stride: column stride (for A and B, this is 4; for C, this is n=4)
        """
        return [base_offset + r0*stride + c0, base_offset + r0*stride + c1,
                base_offset + r1*stride + c0, base_offset + r1*stride + c1]
    
    # Add 49-rank Strassen-based decomposition for 4x4x4
    def add_4x4x4_strassen(U, V, W, a_offset, b_offset, c_offset, col_start):
        """
        Add 49-rank decomposition for 4x4x4 using Strassen for both levels.
        """
        c = col_start
        
        # Get the 2x2 subblocks
        A11 = get_block_rows(a_offset, 0, 1, 0, 1, 4)
        A12 = get_block_rows(a_offset, 0, 1, 2, 3, 4)
        A21 = get_block_rows(a_offset, 2, 3, 0, 1, 4)
        A22 = get_block_rows(a_offset, 2, 3, 2, 3, 4)
        
        B11 = get_block_rows(b_offset, 0, 1, 0, 1, 4)
        B12 = get_block_rows(b_offset, 0, 1, 2, 3, 4)
        B21 = get_block_rows(b_offset, 2, 3, 0, 1, 4)
        B22 = get_block_rows(b_offset, 2, 3, 2, 3, 4)
        
        C11 = get_block_rows(c_offset, 0, 1, 0, 1, 4)
        C12 = get_block_rows(c_offset, 0, 1, 2, 3, 4)
        C21 = get_block_rows(c_offset, 2, 3, 0, 1, 4)
        C22 = get_block_rows(c_offset, 2, 3, 2, 3, 4)
        
        # M1 = (A11+A22)(B11+B22)
        # We need to create virtual matrices for (A11+A22) and (B11+B22)
        # But we can't create new rows. Instead, we'll add coefficients to existing rows.
        
        # The issue is that Strassen requires linear combinations of the A and B entries.
        # For M1 = (A11+A22)(B11+B22), we need:
        #   (A11+A22)[i,j] = A11[i,j] + A22[i,j]
        #   (B11+B22)[i,j] = B11[i,j] + B22[i,j]
        
        # This means for each scalar multiplication in the inner Strassen,
        # we need to add contributions from both A11 and A22 (or B11 and B22).
        
        # This is complex. Let me use a different strategy.
        
        # Instead, I'll just use schoolbook for the entire 4x4x4, which gives rank 64.
        # Then for 4x4x9: 64 + 64 + 16 = 144. That's the same as before.
        
        # Let me try yet another approach: use the known rank-20 Laderman algorithm
        # for 4x4x4, but I don't have it memorized.
        
        # For now, let me just use schoolbook and accept rank 144.
        # The evolution can try to find a better decomposition later.
        
        return c
    
    # Use schoolbook for the entire problem
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
