"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    # Block decomposition using Strassen's 2x2x2 multiplication on 2x2 sub-blocks.
    # n=2, m=4, p=6.
    # Divide A (2x4) into 1x2 blocks of 2x2 matrices: A_left (cols 0-1), A_right (cols 2-3).
    # Divide B (4x6) into 2x3 blocks of 2x2 matrices: B_top (rows 0-1), B_bottom (rows 2-3).
    # C = A * B = [A_left * B_top + A_right * B_bottom] (since n=2, there is only 1 row of 2x2 blocks in C).
    # Wait, C is 2x6.
    # C_{0:2, 0:2} = A_{0:2, 0:2} * B_{0:2, 0:2} + A_{0:2, 2:4} * B_{2:4, 0:2}
    # C_{0:2, 2:4} = A_{0:2, 0:2} * B_{0:2, 2:4} + A_{0:2, 2:4} * B_{2:4, 2:4}
    # C_{0:2, 4:6} = A_{0:2, 0:2} * B_{0:2, 4:6} + A_{0:2, 2:4} * B_{2:4, 4:6}
    
    # Each 2x2 multiplication uses Strassen (rank 7).
    # There are 3 such products for the left part (A_left * B_top_k) and 3 for the right part (A_right * B_bottom_k).
    # Total rank = 3 * 7 + 3 * 7 = 42.
    
    n, m, p = DIMENSIONS
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Strassen coefficients for 2x2 multiplication C = A * B:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We will generate 6 sets of 7 Strassen terms.
    # Set 0: A_left * B_top_0 (contributes to C[:, 0:2])
    # Set 1: A_left * B_top_1 (contributes to C[:, 2:4])
    # Set 2: A_left * B_top_2 (contributes to C[:, 4:6])
    # Set 3: A_right * B_bottom_0 (contributes to C[:, 0:2])
    # Set 4: A_right * B_bottom_1 (contributes to C[:, 2:4])
    # Set 5: A_right * B_bottom_2 (contributes to C[:, 4:6])
    
    def add_strassen_term(set_idx, col_offset, U_row_offsets, V_row_offsets, W_col_offsets, coeffs):
        """
        Adds 7 Strassen terms for a specific 2x2x2 multiplication.
        set_idx: 0-5
        col_offset: base column in U, V, W
        U_row_offsets: list of 4 indices in U corresponding to A00, A01, A10, A11
        V_row_offsets: list of 4 indices in V corresponding to B00, B01, B10, B11
        W_col_offsets: list of 4 indices in W corresponding to C00, C01, C10, C11
        coeffs: not used directly, hard-coded Strassen logic.
        """
        base_col = col_offset
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        c = base_col
        U[U_row_offsets[0]][c] += 1
        U[U_row_offsets[3]][c] += 1
        V[V_row_offsets[0]][c] += 1
        V[V_row_offsets[3]][c] += 1
        W[W_col_offsets[0]][c] += 1
        W[W_col_offsets[3]][c] += 1
        
        # Term 2: M2 = (A10+A11)B00
        c = base_col + 1
        U[U_row_offsets[2]][c] += 1
        U[U_row_offsets[3]][c] += 1
        V[V_row_offsets[0]][c] += 1
        W[W_col_offsets[1]][c] += 1
        W[W_col_offsets[3]][c] += -1
        
        # Term 3: M3 = A00(B10-B11)
        c = base_col + 2
        U[U_row_offsets[0]][c] += 1
        V[V_row_offsets[2]][c] += 1
        V[V_row_offsets[3]][c] += -1
        W[W_col_offsets[1]][c] += 1
        W[W_col_offsets[3]][c] += 1
        
        # Term 4: M4 = A11(B01-B00)
        c = base_col + 3
        U[U_row_offsets[3]][c] += 1
        V[V_row_offsets[1]][c] += 1
        V[V_row_offsets[0]][c] += -1
        W[W_col_offsets[0]][c] += -1
        W[W_col_offsets[2]][c] += 1
        
        # Term 5: M5 = (A00+A01)B11
        c = base_col + 4
        U[U_row_offsets[0]][c] += 1
        U[U_row_offsets[1]][c] += 1
        V[V_row_offsets[3]][c] += 1
        W[W_col_offsets[0]][c] += 1
        W[W_col_offsets[1]][c] += 1
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        c = base_col + 5
        U[U_row_offsets[2]][c] += 1
        U[U_row_offsets[0]][c] += -1
        V[V_row_offsets[0]][c] += 1
        V[V_row_offsets[1]][c] += 1
        W[W_col_offsets[3]][c] += 1
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        c = base_col + 6
        U[U_row_offsets[1]][c] += 1
        U[U_row_offsets[3]][c] += -1
        V[V_row_offsets[2]][c] += 1
        V[V_row_offsets[3]][c] += 1
        W[W_col_offsets[0]][c] += 1

    # Indices mapping:
    # A is 2x4. Flattened U rows:
    # i=0, j=0 -> U[0] (A00 of left)
    # i=0, j=1 -> U[1] (A01 of left)
    # i=1, j=0 -> U[2] (A10 of left)
    # i=1, j=1 -> U[3] (A11 of left)
    # i=0, j=2 -> U[4] (A00 of right)
    # i=0, j=3 -> U[5] (A01 of right)
    # i=1, j=2 -> U[6] (A10 of right)
    # i=1, j=3 -> U[7] (A11 of right)
    
    # U_rows_left = [0, 1, 2, 3] corresponding to A00, A01, A10, A11
    # U_rows_right = [4, 5, 6, 7]
    
    U_left = [0, 1, 2, 3]
    U_right = [4, 5, 6, 7]
    
    # B is 4x6. Flattened V rows: V[j*p+k]
    # Block 0 (B_top_0): rows 0,1 cols 0,1.
    # B00 = V[0*6+0]=V[0], B01=V[1], B10=V[6], B11=V[7]
    # Block 1 (B_top_1): rows 0,1 cols 2,3.
    # B00 = V[0*6+2]=V[2], B01=V[3], B10=V[8], B11=V[9]
    # Block 2 (B_top_2): rows 0,1 cols 4,5.
    # B00 = V[0*6+4]=V[4], B01=V[5], B10=V[10], B11=V[11]
    
    # Block 3 (B_bottom_0): rows 2,3 cols 0,1.
    # B00 = V[2*6+0]=V[12], B01=V[13], B10=V[18], B11=V[19]
    # Block 4 (B_bottom_1): rows 2,3 cols 2,3.
    # B00 = V[2*6+2]=V[14], B01=V[15], B10=V[20], B11=V[21]
    # Block 5 (B_bottom_2): rows 2,3 cols 4,5.
    # B00 = V[2*6+4]=V[16], B01=V[17], B10=V[22], B11=V[23]
    
    V_blocks = [
        [0, 1, 6, 7],   # 0
        [2, 3, 8, 9],   # 1
        [4, 5, 10, 11], # 2
        [12, 13, 18, 19],# 3
        [14, 15, 20, 21],# 4
        [16, 17, 22, 23] # 5
    ]
    
    # C is 2x6. Flattened W rows: W[k*n+i]
    # C00 (col 0): W[0*2+0]=W[0]
    # C01 (col 1): W[0*2+1]=W[1]
    # C10 (col 0): W[0*2+1]? No.
    # Let's trace indices carefully.
    # W index is k*n + i.
    # C is 2x6.
    # C_{i,k} is at W[k*2 + i].
    
    # Block 0 (C[:, 0:2]):
    # C00 (i=0, k=0) -> W[0]
    # C01 (i=0, k=1) -> W[2]
    # C10 (i=1, k=0) -> W[1]
    # C11 (i=1, k=1) -> W[3]
    # W_cols_0 = [0, 2, 1, 3]
    
    # Block 1 (C[:, 2:4]):
    # C00 (i=0, k=2) -> W[4]
    # C01 (i=0, k=3) -> W[6]
    # C10 (i=1, k=2) -> W[5]
    # C11 (i=1, k=3) -> W[7]
    # W_cols_1 = [4, 6, 5, 7]
    
    # Block 2 (C[:, 4:6]):
    # C00 (i=0, k=4) -> W[8]
    # C01 (i=0, k=5) -> W[10]
    # C10 (i=1, k=4) -> W[9]
    # C11 (i=1, k=5) -> W[11]
    # W_cols_2 = [8, 10, 9, 11]
    
    W_blocks = [
        [0, 2, 1, 3],   # 0
        [4, 6, 5, 7],   # 1
        [8, 10, 9, 11], # 2
        [0, 2, 1, 3],   # 3 (contributes to same C block as 0)
        [4, 6, 5, 7],   # 4 (contributes to same C block as 1)
        [8, 10, 9, 11]  # 5 (contributes to same C block as 2)
    ]
    
    # Set 0: A_left * B_top_0
    add_strassen_term(0, 0, U_left, V_blocks[0], W_blocks[0], None)
    # Set 1: A_left * B_top_1
    add_strassen_term(1, 7, U_left, V_blocks[1], W_blocks[1], None)
    # Set 2: A_left * B_top_2
    add_strassen_term(2, 14, U_left, V_blocks[2], W_blocks[2], None)
    # Set 3: A_right * B_bottom_0
    add_strassen_term(3, 21, U_right, V_blocks[3], W_blocks[3], None)
    # Set 4: A_right * B_bottom_1
    add_strassen_term(4, 28, U_right, V_blocks[4], W_blocks[4], None)
    # Set 5: A_right * B_bottom_2
    add_strassen_term(5, 35, U_right, V_blocks[5], W_blocks[5], None)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
