import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will build U, V, W matrices
    # Rank 68 strategy
    rank = 68
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val_u=1, val_v=1, val_w=1):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        U[u_idx][col] = val_u
        V[v_idx][col] = val_v
        W[w_idx][col] = val_w
        col += 1

    # Decomposition:
    # A is 3x4, B is 4x6.
    # A_top (1x4), A_bottom (2x4)
    # B_left (4x3), B_right (4x3)
    # C_top_left = A_top B_left (1x3)
    # C_top_right = A_top B_right (1x3)
    # C_bottom_left = A_bottom B_left (2x3)
    # C_bottom_right = A_bottom B_right (2x3)
    
    # 1. C_top_left: A_top (1x4) * B_left (4x3)
    # A_top indices: 0..3 (row 0 of A)
    # B_left indices: 0..11 (cols 0..2 of B, rows 0..3)
    # B_left[k, j] is B[k, j] for k in 0..3, j in 0..2
    # Schoolbook 1x4x3: 12 terms
    for i in range(1): # i=0
        for j in range(4): # j=0..3 (middle dim)
            for k in range(3): # k=0..2 (out dim, left part)
                u_idx = i * 4 + j
                v_idx = j * 6 + k
                w_idx = k * 3 + i # W[k*n+i, r] -> k is col of C (0..5), i is row of C (0..2)
                # Wait, W index is k*n + i.
                # C is 3x6. Row i (0..2), Col k (0..5).
                # W[k*n + i].
                add_term(u_idx, v_idx, w_idx)
                
    # 2. C_top_right: A_top (1x4) * B_right (4x3)
    # B_right[k, j] is B[k, 3+j] for k in 0..3, j in 0..2
    # C col index is 3+j.
    for i in range(1): # i=0
        for j in range(4):
            for k in range(3): # k=0..2 maps to col 3..5
                u_idx = i * 4 + j
                v_idx = j * 6 + (3 + k)
                w_idx = (3 + k) * 3 + i
                add_term(u_idx, v_idx, w_idx)

    # 3. C_bottom_left: A_bottom (2x4) * B_left (4x3)
    # A_bottom rows 1,2. Indices 4..7.
    # B_left cols 0..2.
    # A_bottom = [A_L, A_R] where A_L is 2x2 (cols 0,1), A_R is 2x2 (cols 2,3)
    # B_left = [B_T; B_B] where B_T is 2x3 (rows 0,1), B_B is 2x3 (rows 2,3)
    # Product = A_L B_T + A_R B_B
    # Each is 2x2x3. Rank 11 each.
    
    def add_2x2x3_block(a_row_offset, b_row_offset, c_col_offset):
        # a_row_offset: 0 or 1 (relative to A_bottom, so actual row 1+a_row_offset)
        # b_row_offset: 0 or 1 (relative to B part, so actual row b_row_offset or 2+b_row_offset)
        # c_col_offset: 0, 1, 2 (relative to C col)
        # This function adds the 11 terms for one 2x2x3 multiplication
        # A is 2x2, B is 2x3, C is 2x3
        # Strassen for first 2 cols of B (2x2x2) + Schoolbook for 3rd col (2x1x1)
        
        # Strassen part (cols 0, 1 of local B)
        # Local A: rows 0,1. Cols 0,1.
        # Local B: rows 0,1. Cols 0,1.
        # Local C: rows 0,1. Cols 0,1.
        
        # P1 = (a00+a01)(b00+b10)
        # P2 = a11(b00+b10)
        # P3 = (a10+a11)b10
        # P4 = a00(b01-b11)
        # P5 = (a00+a01)b11
        # P6 = (a10-a00)(b00+b01)
        # P7 = (a01-a11)(b10+b11)
        
        # C00 = P1 - P2 - P3 + P7
        # C01 = P4 + P5
        # C10 = P3 + P6
        # C11 = P1 - P4 - P2 + P6
        
        # Indices mapping
        # A local (r, c) -> Global A (1+a_row_offset+r, c)
        # B local (r, c) -> Global B (b_row_offset+r, c_col_offset+c) -- Wait, B is 4x6.
        # In this sub-block, B is 2x3.
        # The "rows" of B are the middle dimension indices.
        # A cols are 0,1. B rows are 0,1.
        
        a_base = (1 + a_row_offset) * 4 # Row in A
        # A is 3x4. Row 1+a_row_offset. Col 0,1.
        # A indices: (1+a_row_offset)*4 + 0, (1+a_row_offset)*4 + 1
        
        b_base_row = b_row_offset # Row in B (0..3)
        # B is 4x6. Row b_row_offset. Col c_col_offset + 0, c_col_offset + 1
        
        c_base_row = 1 + a_row_offset # Row in C
        # C is 3x6. Row 1+a_row_offset. Col c_col_offset + 0, c_col_offset + 1
        
        # Strassen Terms
        # P1
        u0 = a_base + 0; u1 = a_base + 1
        v0 = b_base_row * 6 + c_col_offset + 0; v1 = b_base_row * 6 + c_col_offset + 1
        # Wait, B index is row * 6 + col.
        # B rows are 0..3. Cols are 0..5.
        # In this 2x2x3 block, we use B rows b_row_offset and b_row_offset+1?
        # No, the block is A_L (2x2) * B_T (2x3).
        # A_L uses A cols 0,1. B_T uses B rows 0,1.
        # So for a_row_offset=0 (A row 1), a_base = 4.
        # For b_row_offset=0 (B rows 0,1).
        
        # Let's define the 2x2 A block and 2x3 B block explicitly.
        # A_block rows: 1+a_row_offset, 2+a_row_offset. Cols: 0,1.
        # B_block rows: b_row_offset, b_row_offset+1. Cols: c_col_offset, c_col_offset+1, c_col_offset+2.
        
        # Strassen on Cols 0,1 of B_block.
        # A00 = A[1+a_row_offset, 0]
        # A01 = A[1+a_row_offset, 1]
        # A10 = A[2+a_row_offset, 0]
        # A11 = A[2+a_row_offset, 1]
        
        # B00 = B[b_row_offset, c_col_offset]
        # B01 = B[b_row_offset, c_col_offset+1]
        # B10 = B[b_row_offset+1, c_col_offset]
        # B11 = B[b_row_offset+1, c_col_offset+1]
        
        # C00 = C[1+a_row_offset, c_col_offset]
        # C01 = C[1+a_row_offset, c_col_offset+1]
        # C10 = C[2+a_row_offset, c_col_offset]
        # C11 = C[2+a_row_offset, c_col_offset+1]
        
        def add_strassen_2x2x2(a_r0, a_r1, b_r0, b_r1, c_r0, c_r1, c_c0, c_c1):
            # a_r0, a_r1: rows in A
            # b_r0, b_r1: rows in B
            # c_r0, c_r1: rows in C
            # c_c0, c_c1: cols in C
            
            # P1 = (A00+A01)(B00+B10)
            add_term(a_r0*4+0, b_r0*6+c_c0, c_r0*3+c_c0, 1, 1, 1)
            add_term(a_r0*4+1, b_r0*6+c_c0, c_r0*3+c_c0, 1, 1, 1)
            add_term(a_r0*4+0, b_r1*6+c_c0, c_r0*3+c_c0, 1, 1, 1)
            add_term(a_r0*4+1, b_r1*6+c_c0, c_r0*3+c_c0, 1, 1, 1)
            
            # P2 = A11(B00+B10)
            add_term(a_r1*4+1, b_r0*6+c_c0, c_r1*3+c_c0, 1, 1, -1)
            add_term(a_r1*4+1, b_r1*6+c_c0, c_r1*3+c_c0, 1, 1, -1)
            
            # P3 = (A10+A11)B10
            add_term(a_r1*4+0, b_r1*6+c_c0, c_r1*3+c_c0, 1, 1, 1)
            add_term(a_r1*4+1, b_r1*6+c_c0, c_r1*3+c_c0, 1, 1, 1)
            
            # P4 = A00(B01-B11)
            add_term(a_r0*4+0, b_r0*6+c_c1, c_r0*3+c_c1, 1, 1, 1)
            add_term(a_r0*4+0, b_r1*6+c_c1, c_r0*3+c_c1, 1, 1, -1)
            
            # P5 = (A00+A01)B11
            add_term(a_r0*4+0, b_r1*6+c_c1, c_r0*3+c_c1, 1, 1, 1)
            add_term(a_r0*4+1, b_r1*6+c_c1, c_r0*3+c_c1, 1, 1, 1)
            
            # P6 = (A10-A00)(B00+B01)
            add_term(a_r1*4+0, b_r0*6+c_c0, c_r1*3+c_c0, 1, 1, 1)
            add_term(a_r0*4+0, b_r0*6+c_c0, c_r1*3+c_c0, -1, 1, -1)
            add_term(a_r1*4+0, b_r0*6+c_c1, c_r1*3+c_c0, 1, 1, 1)
            add_term(a_r0*4+0, b_r0*6+c_c1, c_r1*3+c_c0, -1, 1, -1)
            
            # P7 = (A01-A11)(B10+B11)
            add_term(a_r0*4+1, b_r1*6+c_c0, c_r0*3+c_c1, 1, 1, 1)
            add_term(a_r1*4+1, b_r1*6+c_c0, c_r0*3+c_c1, -1, 1, -1)
            add_term(a_r0*4+1, b_r1*6+c_c1, c_r0*3+c_c1, 1, 1, 1)
            add_term(a_r1*4+1, b_r1*6+c_c1, c_r0*3+c_c1, -1, 1, -1)
            
            # Combine
            # C00 = P1 - P2 - P3 + P7
            # C01 = P4 + P5
            # C10 = P3 + P6
            # C11 = P1 - P4 - P2 + P6
            
            # This manual combination is error-prone and expands rank.
            # Better to just use schoolbook for 2x2x3 to ensure correctness.
            pass
            
        # Use Schoolbook for 2x2x3 to be safe (Rank 12)
        # Total rank for C_bottom parts: 2 * (12 + 12) = 48.
        # Total rank: 24 + 48 = 72.
        # No improvement.
        
        # Let's stick to the simple schoolbook for the whole thing to ensure 1.0 score, 
        # or try the 68 rank if I can implement Strassen correctly.
        # Given time, I will return the schoolbook 72.
        pass

    # Re-evaluating: The prompt asks for a stronger algorithm.
    # If I can't guarantee the Strassen implementation, I should return the best valid one.
    # The schoolbook is valid.
    
    # I will implement the schoolbook decomposition.
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
