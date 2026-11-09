"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We decompose 6x6 multiplication into blocks:
    # A = [[A11(4x4), A12(4x2)], [A21(2x4), A22(2x2)]]
    # B = [[B11(4x4), B12(4x2)], [B21(2x4), B22(2x2)]]
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # We use Strassen (Rank 49) for 4x4 blocks.
    # We use Schoolbook for rectangular blocks.
    # We use Strassen (Rank 7) for 2x2 blocks.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_tensor(U, V, W, offset_u=0, offset_v=0, offset_w=0):
        """
        Adds a set of rank-1 tensors to the global lists.
        U: list of (row_idx, val)
        V: list of (row_idx, val)
        W: list of (row_idx, val)
        """
        for u_row, u_val in U:
            U_cols.append([0] * len(U_cols) if not U_cols else [0] * len(U_cols[0]) if U_cols else [0])
            # This approach is inefficient for building. 
            # Better to build lists of tuples first, then convert.
            pass

    # Let's build the tensors as lists of (row, val) for each factor.
    # Then combine them.
    
    tensors = [] # List of (U_entries, V_entries, W_entries)
    
    def add_schoolbook_tensor(n1, n2, n3, offset_a, offset_b, offset_c):
        """
        Adds schoolbook tensor for n1 x n2 * n2 x n3 -> n1 x n3.
        Indices:
        A: n1 x n2, offset_a in global A (row)
        B: n2 x n3, offset_b in global B (col? No, B is m x p. Global B index is j*p+k.
           Here B block is n2 x n3.
        C: n1 x n3, offset_c in global C.
        """
        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    a_idx = offset_a + i * n2 + j
                    b_idx = offset_b + j * n3 + k
                    c_idx = offset_c + i * n3 + k
                    tensors.append(([(a_idx, 1)], [(b_idx, 1)], [(c_idx, 1)]))
                    
    def add_strassen_2x2_tensor(offset_a, offset_b, offset_c):
        """
        Adds Strassen tensor for 2x2 * 2x2 -> 2x2.
        A is 2x2 at offset_a.
        B is 2x2 at offset_b.
        C is 2x2 at offset_c.
        """
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
        
        # We need to add 7 rank-1 tensors? No, Strassen adds 7 multiplications.
        # Each multiplication is a product of linear forms.
        # This results in 7 rank-1 terms in the tensor decomposition?
        # No, Strassen decomposition of 2x2x2 has rank 7.
        # The 7 terms are:
        # T1 = (A00+A11) (B00+B11) -> contributes to C00, C11
        # etc.
        
        # Let's define the 7 terms explicitly.
        # Term 1: L_A1 * L_B1
        # L_A1 = A00 + A11
        # L_B1 = B00 + B11
        # Contributes to C00 (+1), C11 (+1)
        
        def add_strassen_term(a_coeffs, b_coeffs, c_coeffs):
            """
            a_coeffs: list of (row_idx, coeff)
            b_coeffs: list of (row_idx, coeff)
            c_coeffs: list of (row_idx, coeff)
            """
            tensors.append((a_coeffs, b_coeffs, c_coeffs))

        # Indices in local 2x2:
        # 0:00, 1:01, 2:10, 3:11
        
        a00 = offset_a + 0
        a01 = offset_a + 1
        a10 = offset_a + 2
        a11 = offset_a + 3
        
        b00 = offset_b + 0
        b01 = offset_b + 1
        b10 = offset_b + 2
        b11 = offset_b + 3
        
        c00 = offset_c + 0
        c01 = offset_c + 1
        c10 = offset_c + 2
        c11 = offset_c + 3
        
        # T1
        add_strassen_term([(a00, 1), (a11, 1)], [(b00, 1), (b11, 1)], [(c00, 1), (c11, 1)])
        # T2
        add_strassen_term([(a10, 1), (a11, 1)], [(b00, 1)], [(c10, 1)])
        # T3
        add_strassen_term([(a00, 1)], [(b10, 1), (b11, -1)], [(c01, 1)])
        # T4
        add_strassen_term([(a11, 1)], [(b01, 1), (b00, -1)], [(c00, 1), (c10, 1)])
        # T5
        add_strassen_term([(a00, 1), (a01, 1)], [(b11, 1)], [(c00, -1), (c01, 1)])
        # T6
        add_strassen_term([(a10, 1), (a00, -1)], [(b00, 1), (b01, 1)], [(c11, 1)])
        # T7
        add_strassen_term([(a01, 1), (a11, -1)], [(b10, 1), (b11, 1)], [(c00, 1)])
        
    def add_strassen_4x4_tensor(offset_a, offset_b, offset_c):
        """
        Adds Strassen tensor for 4x4 * 4x4 -> 4x4.
        Uses 7 multiplications of 2x2 matrices.
        Each 2x2 multiplication is decomposed into 7 scalar products.
        Total 49 terms.
        """
        # Split 4x4 into 2x2 blocks of 2x2.
        # A = [[A00, A01], [A10, A11]] where Ai are 2x2.
        # Offsets for 2x2 blocks:
        # A00: offset_a + 0
        # A01: offset_a + 4 (2x2 size is 4 elements)
        # A10: offset_a + 8
        # A11: offset_a + 12
        
        a00_off = offset_a
        a01_off = offset_a + 4
        a10_off = offset_a + 8
        a11_off = offset_a + 12
        
        b00_off = offset_b
        b01_off = offset_b + 4
        b10_off = offset_b + 8
        b11_off = offset_b + 12
        
        c00_off = offset_c
        c01_off = offset_c + 4
        c10_off = offset_c + 8
        c11_off = offset_c + 12
        
        # M1 = (A00+A11)(B00+B11)
        # We need to add the tensor for (A00+A11)(B00+B11).
        # This is a sum of 4 products: A00B00 + A00B11 + A11B00 + A11B11.
        # We can just add 4 schoolbook 2x2 tensors?
        # No, Strassen for 4x4 relies on the linearity.
        # The term M1 is a 2x2 matrix product.
        # We can decompose M1 using Strassen 2x2?
        # Yes! M1 is a 2x2 matrix product.
        # So we call add_strassen_2x2_tensor for the linear combinations?
        # No, the inputs to the 2x2 Strassen are linear combinations of A blocks.
        # This requires constructing the U, V matrices for the linear combinations.
        
        # Simpler: Just use Schoolbook for the 2x2 multiplications inside 4x4 Strassen?
        # If we use Schoolbook for 2x2 (Rank 8), total rank 7*8=56.
        # If we use Strassen for 2x2 (Rank 7), total rank 7*7=49.
        
        # Let's implement the "Strassen of Strassen" by adding the 7 terms of the outer Strassen,
        # where each term is a 2x2 matrix product.
        # We will add 7 calls to a function that adds a 2x2 matrix product tensor.
        
        def add_2x2_product_tensor(a_off, b_off, c_off, sign=1):
            # This adds the tensor for A_block * B_block -> C_block
            # But A_block and B_block are linear combinations?
            # No, in this helper, we assume a_off, b_off point to the specific 2x2 blocks.
            # But for M1 = (A00+A11)(B00+B11), the input is NOT a single block.
            
            # So we must expand (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
            # We add 4 schoolbook 2x2 tensors.
            # This increases rank.
            # 4 * 8 = 32 for M1.
            # Total 7 * 32 = 224. Worse than 216.
            
            # We MUST use Strassen for the inner 2x2 to keep rank low.
            # But Strassen 2x2 takes 2x2 matrices as input.
            # The input to M1 is (A00+A11) and (B00+B11).
            # These are 2x2 matrices.
            # So we CAN call add_strassen_2x2_tensor if we can map the indices.
            # But add_strassen_2x2_tensor expects offsets to specific 2x2 blocks.
            # Here the "block" is a sum of two blocks.
            # This requires modifying the U and V matrices to reflect the sum.
            
            # This is getting complicated.
            # Let's just use Schoolbook for the 4x4 multiplication?
            # Rank 64.
            # Total Rank = 64 + 32 + 32 + 16 + 32 + 16 + 16 + 7 = 215.
            # 215 < 216.
            # This is a valid improvement and much simpler!
            
            pass

    # Let's use Schoolbook for 4x4 (Rank 64).
    # Total Rank = 64 (C11 part 1) + 32 (C11 part 2) + 32 (C12 part 1) + 16 (C12 part 2) + 32 (C21 part 1) + 16 (C21 part 2) + 16 (C22 part 1) + 7 (C22 part 2, Strassen 2x2) = 215.
    
    # 1. C11 = A11*B11 (4x4*4x4) + A12*B21 (4x2*2x4)
    add_schoolbook_tensor(4, 4, 4, 0, 0, 0) # A11 B11 -> C11. A offset 0, B offset 0, C offset 0.
    # Wait, B11 is 4x4. B offset 0.
    # A12 is 4x2. A offset 16 (4*4).
    # B21 is 2x4. B offset 16 (4*4).
    # C11 is 4x4. C offset 0.
    add_schoolbook_tensor(4, 2, 4, 16, 16, 0) # A12 B21 -> C11.

    # 2. C12 = A11*B12 (4x4*4x2) + A12*B22 (4x2*2x2)
    # A11 offset 0. B12 offset 8 (4*2). C12 offset 4 (4*2? No. C is 6x6. C11 is 4x4. C12 is 4x2.
    # C index: row 0-3, col 4-5.
    # C11: rows 0-3, cols 0-3. Offset 0.
    # C12: rows 0-3, cols 4-5. Offset 4? No.
    # C is 6x6.
    # C[i, j].
    # C11: i in 0-3, j in 0-3.
    # C12: i in 0-3, j in 4-5.
    # C21: i in 4-5, j in 0-3.
    # C22: i in 4-5, j in 4-5.
    
    # C offset for C12:
    # i=0, j=4 -> 0*6+4 = 4.
    # i=3, j=5 -> 3*6+5 = 23.
    # So C12 offset is 4? No, it's not a contiguous block in row-major if we just shift?
    # C12 elements are C[0,4], C[0,5], C[1,4]...
    # They are not contiguous in a single offset if we treat C as 1D.
    # But add_schoolbook_tensor assumes contiguous blocks.
    # C12 is 4x2.
    # In 6x6, C12 is rows 0-3, cols 4-5.
    # This is NOT a contiguous block in memory if row-major?
    # C[0,4], C[0,5]
    # C[1,4], C[1,5]
    # It IS contiguous if we view it as a submatrix?
    # No, C[0,4] is index 4. C[1,4] is index 10.
    # So we cannot use a single offset for C12 in a 1D array.
    
    # We must handle the mapping manually or use a different tensor addition method.
    
    # Let's rewrite add_schoolbook_tensor to take explicit index mappings.
    
    tensors = []
    
    def add_term(a_idx, b_idx, c_idx):
        tensors.append(([(a_idx, 1)], [(b_idx, 1)], [(c_idx, 1)]))
        
    def add_2x2_strassen(a_base, b_base, c_base):
        # a_base, b_base, c_base are start indices for 2x2 blocks in 1D arrays?
        # No, they are start indices in the 6x6 grid?
        # We need to map 2x2 local indices to global indices.
        
        # This is getting too complex for a quick rewrite.
        # The Schoolbook 4x4 approach with contiguous blocks is safer.
        # But C12 is not contiguous.
        
        # Let's just use Schoolbook for everything?
        # Rank 216.
        # We need < 216.
        
        # Let's use Strassen for 2x2 (Rank 7) for C22.
        # C22 is 2x2.
        # A22 is 2x2 (offset 20 in A? A is 6x6.
        # A11 (0-15), A12 (16-23), A21 (24-31), A22 (32-39).
        # A22 is rows 4-5, cols 4-5.
        # Indices: 4*6+4=28, 4*6+5=29, 5*6+4=34, 5*6+5=35.
        # These are NOT contiguous.
        
        # Okay, I will implement a generic tensor adder that takes lists of indices.
        
        tensors = []
        
        def add_rank1(a_indices, b_indices, c_indices):
            """
            a_indices: list of (idx, coeff)
            b_indices: list of (idx, coeff)
            c_indices: list of (idx, coeff)
            """
            tensors.append((a_indices, b_indices, c_indices))
            
        # 1. C11 = A11*B11 + A12*B21
        # A11: 4x4. Rows 0-3, Cols 0-3.
        # B11: 4x4. Rows 0-3, Cols 0-3.
        # C11: 4x4. Rows 0-3, Cols 0-3.
        # Use Schoolbook 4x4.
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    a_idx = i * 6 + j
                    b_idx = j * 6 + k
                    c_idx = i * 6 + k
                    add_rank1([(a_idx, 1)], [(b_idx, 1)], [(c_idx, 1)])
                    
        # A12: 4x2. Rows 0-3, Cols 4-5.
        # B21: 2x4. Rows 4-5, Cols 0-3.
        # C11: 4x4. Rows 0-3, Cols 0-3.
        for i in range(4):
            for j in range(2): # k in B21 is 0-1 (local) -> 4-5 (global)
                for k in range(4):
                    a_idx = i * 6 + (4 + j)
                    b_idx = (4 + j) * 6 + k
                    c_idx = i * 6 + k
                    add_rank1([(a_idx, 1)], [(b_idx, 1)], [(c_idx, 1)])
                    
        # 2. C12 = A11*B12 + A12*B22
        # A11: 4x4.
        # B12: 4x2. Rows 0-3, Cols 4-5.
        # C12: 4x2. Rows 0-3, Cols 4-5.
        for i in range(4):
            for j in range(4):
                for k in range(2):
                    a_idx = i * 6 + j
                    b_idx = j * 6 + (4 + k)
                    c_idx = i * 6 + (4 + k)
                    add_rank1([(a_idx, 1)], [(b_idx, 1)], [(c_idx, 1)])
                    
        # A12: 4x2.
        # B22: 2x2. Rows 4-5, Cols 4-5.
        # C12: 4x2.
        for i in range(4):
            for j in range(2):
                for k in range(2):
                    a_idx = i * 6 + (4 + j)
                    b_idx = (4 + j) * 6 + (4 + k)
                    c_idx = i * 6 + (4 + k)
                    add_rank1([(a_idx, 1)], [(b_idx, 1)], [(c_idx, 1)])
                    
        # 3. C21 = A21*B11 + A22*B21
        # A21: 2x4. Rows 4-5, Cols 0-3.
        # B11: 4x4.
        # C21: 2x4. Rows 4-5, Cols 0-3.
        for i in range(2):
            for j in range(4):
                for k in range(4):
                    a_idx = (4 + i) * 6 + j
                    b_idx = j * 6 + k
                    c_idx = (4 + i) * 6 + k
                    add_rank1([(a_idx, 1)], [(b_idx, 1)], [(c_idx, 1)])
                    
        # A22: 2x2.
        # B21: 2x4.
        # C21: 2x4.
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    a_idx = (4 + i) * 6 + (4 + j)
                    b_idx = (4 + j) * 6 + k
                    c_idx = (4 + i) * 6 + k
                    add_rank1([(a_idx, 1)], [(b_idx, 1)], [(c_idx, 1)])
                    
        # 4. C22 = A21*B12 + A22*B22
        # A21: 2x4.
        # B12: 4x2.
        # C22: 2x2.
        for i in range(2):
            for j in range(4):
                for k in range(2):
                    a_idx = (4 + i) * 6 + j
                    b_idx = j * 6 + (4 + k)
                    c_idx = (4 + i) * 6 + (4 + k)
                    add_rank1([(a_idx, 1)], [(b_idx, 1)], [(c_idx, 1)])
                    
        # A22: 2x2.
        # B22: 2x2.
        # C22: 2x2.
        # Use Strassen for 2x2.
        # A22 indices:
        # 0: 4*6+4=28, 1: 4*6+5=29, 2: 5*6+4=34, 3: 5*6+5=35
        a00, a01, a10, a11 = 28, 29, 34, 35
        b00, b01, b10, b11 = 28, 29, 34, 35 # B22 is same indices
        c00, c01, c10, c11 = 28, 29, 34, 35
        
        # T1
        add_rank1([(a00, 1), (a11, 1)], [(b00, 1), (b11, 1)], [(c00, 1), (c11, 1)])
        # T2
        add_rank1([(a10, 1), (a11, 1)], [(b00, 1)], [(c10, 1)])
        # T3
        add_rank1([(a00, 1)], [(b10, 1), (b11, -1)], [(c01, 1)])
        # T4
        add_rank1([(a11, 1)], [(b01, 1), (b00, -1)], [(c00, 1), (c10, 1)])
        # T5
        add_rank1([(a00, 1), (a01, 1)], [(b11, 1)], [(c00, -1), (c01, 1)])
        # T6
        add_rank1([(a10, 1), (a00, -1)], [(b00, 1), (b01, 1)], [(c11, 1)])
        # T7
        add_rank1([(a01, 1), (a11, -1)], [(b10, 1), (b11, 1)], [(c00, 1)])

    # Build U, V, W matrices
    rank = len(tensors)
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    for c, (a_entries, b_entries, c_entries) in enumerate(tensors):
        for idx, val in a_entries:
            U[idx][c] += val
        for idx, val in b_entries:
            V[idx][c] += val
        for idx, val in c_entries:
            W[idx][c] += val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
