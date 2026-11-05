"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    # Block-structured decomposition exploiting Strassen for 2x2x2 blocks.
    # Split 2x4x5 into two 2x2x5 blocks (for each row-pair of A).
    # Each 2x2x5 = two 2x2x2 (Strassen, rank 7 each) + one 2x2x1 (schoolbook, rank 4).
    # Total rank = 2 * (7 + 7 + 4) = 36.
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # We'll build U, V, W as lists of lists.
    # U has shape (n*m, rank) = (8, 36)
    # V has shape (m*p, rank) = (20, 36)
    # W has shape (p*n, rank) = (10, 36)
    
    total_rank = 36
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # For each block i (i=0,1) corresponding to rows i*2..i*2+1 of A
    for i_block in range(2):
        # A_block is 2x2 (rows i_block*2 to i_block*2+1, all 4 cols of A)
        # B_block is 2x5 (rows i_block*2 to i_block*2+1 of B, all 5 cols)
        
        # Split B_block into B_4 (2x4) and B_1 (2x1)
        # B_4 is further split into B_4a (2x2, cols 0-1) and B_4b (2x2, cols 2-3)
        
        # --- Strassen for A_block * B_4a (2x2 * 2x2) ---
        # Strassen uses 7 multiplications.
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # For tensor decomposition, each Strassen multiplication corresponds to a rank-1 term.
        # We need to map the Strassen factors to U, V, W entries.
        
        # A_block row indices: i_block*2, i_block*2+1
        # A_block col indices: 0, 1, 2, 3 (but for B_4a, cols 0,1; for B_4b, cols 2,3)
        # B_block row indices: i_block*2, i_block*2+1
        # B_4a col indices: 0, 1
        # B_4b col indices: 2, 3
        # B_1 col index: 4
        
        def add_strassen(col_start, b_col_offset):
            """Add Strassen decomposition for A_block * B_subblock (2x2 * 2x2).
            b_col_offset: which columns of B to use (0 for B_4a, 2 for B_4b).
            Returns next column index.
            """
            nonlocal col
            a_row0 = i_block * 2
            a_row1 = i_block * 2 + 1
            b_row0 = i_block * 2
            b_row1 = i_block * 2 + 1
            
            # Strassen factors for A:
            # A00 = A[a_row0, 0], A01 = A[a_row0, 1], A10 = A[a_row1, 0], A11 = A[a_row1, 1]
            # But wait, A is 2x4, so A_block is 2x2 using cols 0,1 for the "A part"
            # Actually, in the tensor decomposition, A is the left matrix (2x4), B is right (4x5).
            # The "block structure" here is: we're decomposing the multiplication tensor.
            # The Strassen decomposition applies to 2x2 * 2x2 subproblems.
            # 
            # Let me reconsider the structure.
            # We have A (2x4) * B (4x5) -> C (2x5).
            # We split A into two 2x2 blocks horizontally: A_left (2x2, cols 0-1), A_right (2x2, cols 2-3).
            # We split B into two 2x5 blocks vertically: B_top (2x5, rows 0-1), B_bot (2x5, rows 2-3).
            # Then C = A_left * B_top + A_right * B_bot.
            # Each term is 2x2 * 2x5 = 2x5.
            # 
            # For each 2x2 * 2x5, we can further split B into 2x2 + 2x2 + 2x1:
            # B_sub = [B_4a (2x2), B_4b (2x2), B_1 (2x1)]
            # Then A_sub * B_sub = A_sub * B_4a + A_sub * B_4b + A_sub * B_1
            # A_sub * B_4a: 2x2 * 2x2 -> use Strassen (rank 7)
            # A_sub * B_4b: 2x2 * 2x2 -> use Strassen (rank 7)
            # A_sub * B_1: 2x2 * 2x1 -> use schoolbook (rank 4)
            # Total per 2x2 * 2x5: 18
            # Two such blocks: 36
            #
            # Wait, this is different from what I described earlier. Let me redo.
            #
            # Actually, the original plan was:
            # Split A into A_1, A_2 (each 2x2, by rows? No, A is 2x4, so split by columns).
            # A = [A_left | A_right] where A_left is 2x2 (cols 0-1), A_right is 2x2 (cols 2-3).
            # B = [B_top; B_bot] where B_top is 2x5 (rows 0-1), B_bot is 2x5 (rows 2-3).
            # C = A_left * B_top + A_right * B_bot.
            #
            # For A_left * B_top (2x2 * 2x5):
            #   Split B_top into B_top_left (2x2, cols 0-1), B_top_mid (2x2, cols 2-3), B_top_right (2x1, col 4).
            #   A_left * B_top_left: Strassen (rank 7)
            #   A_left * B_top_mid: Strassen (rank 7)
            #   A_left * B_top_right: schoolbook (rank 4)
            #   Total: 18
            #
            # Similarly for A_right * B_bot: 18
            # Total: 36
            #
            # So the block index is over (A_left/A_right) x (B_top/B_bot).
            # Let me redefine the loop.
            
            # This function is called with:
            # - a_col_offset: which columns of A to use (0 for A_left, 2 for A_right)
            # - b_row_offset: which rows of B to use (0 for B_top, 2 for B_bot)
            # - b_col_offset: which columns of B_sub to use (0, 2, or 4)
            # - is_strassen: True for 2x2*2x2, False for 2x2*2x1
            
            pass  # Placeholder, will implement properly below
        
        # Let me restart with a clearer structure.
    
    # Clear and rebuild
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    col = 0
    
    # Strassen coefficients for 2x2*2x2
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_A_factors = [
        [1, 1, 1, 1],  # M1: A00+A11
        [0, 0, 1, 1],  # M2: A10+A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00+A01
        [-1, 0, 1, 0], # M6: A10-A00
        [0, 1, 0, -1], # M7: A01-A11
    ]
    strassen_B_factors = [
        [1, 0, 0, 1],  # M1: B00+B11
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: B01-B11
        [0, 0, 1, -1], # M4: B10-B00
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00+B01
        [0, 0, 1, 1],  # M7: B10+B11
    ]
    # C accumulation: C[row][col] = sum over m of coeff[m] * M_m
    # C00 = M1 + M4 - M5 + M7 -> [1, 0, 0, 1, -1, 0, 1]
    # C01 = M3 + M5 -> [0, 0, 1, 0, 1, 0, 0]
    # C10 = M2 + M4 -> [0, 1, 0, 1, 0, 0, 0]
    # C11 = M1 - M2 + M3 + M6 -> [1, -1, 1, 0, 0, 1, 0]
    strassen_C_coeffs = [
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0],  # C11
    ]
    
    for a_col_off in [0, 2]:  # A_left, A_right
        for b_row_off in [0, 2]:  # B_top, B_bot
            # 2x2 * 2x5 block
            # Split into 2x2*2x2 (cols 0-1), 2x2*2x2 (cols 2-3), 2x2*2x1 (col 4)
            for b_col_off in [0, 2]:  # Two Strassen blocks
                for s in range(7):
                    c = col + s
                    # U: A factors
                    # A is 2x4, we use rows 0,1 and cols a_col_off, a_col_off+1
                    # A_local[0,0] = A[0, a_col_off], A_local[0,1] = A[0, a_col_off+1]
                    # A_local[1,0] = A[1, a_col_off], A_local[1,1] = A[1, a_col_off+1]
                    for r in range(2):
                        for cc in range(2):
                            U[r * m + a_col_off + cc][c] += strassen_A_factors[s][r * 2 + cc]
                    
                    # V: B factors
                    # B is 4x5, we use rows b_row_off, b_row_off+1 and cols b_col_off, b_col_off+1
                    # B_local[0,0] = B[b_row_off, b_col_off], B_local[0,1] = B[b_row_off, b_col_off+1]
                    # B_local[1,0] = B[b_row_off+1, b_col_off], B_local[1,1] = B[b_row_off+1, b_col_off+1]
                    for r in range(2):
                        for cc in range(2):
                            V[(b_row_off + r) * p + b_col_off + cc][c] += strassen_B_factors[s][r * 2 + cc]
                    
                    # W: C coefficients
                    # C is 2x5, we use rows 0,1 and cols b_col_off, b_col_off+1
                    # C_local[0,0] = C[0, b_col_off], C_local[0,1] = C[0, b_col_off+1]
                    # C_local[1,0] = C[1, b_col_off], C_local[1,1] = C[1, b_col_off+1]
                    for r in range(2):
                        for cc in range(2):
                            # W index: k*n + i where k is col of B (= b_col_off+cc), i is row of C (= r)
                            W[(b_col_off + cc) * n + r][c] += strassen_C_coeffs[r * 2 + cc][s]
            
            # Schoolbook for 2x2 * 2x1 (col 4)
            for s in range(4):
                c = col + s
                a_r = s // 2  # row of A (0 or 1)
                a_c = s % 2   # col of A (0 or 1)
                b_r = a_r     # row of B
                b_c = 0       # col of B (always 0 for 2x1)
                
                U[a_r * m + a_col_off + a_c][c] = 1
                V[(b_row_off + b_r) * p + 4][c] = 1
                W[4 * n + a_r][c] = 1  # k=4, i=a_r
            
            col += 4
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
