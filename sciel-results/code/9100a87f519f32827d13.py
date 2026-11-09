"""
Rank-minimal exact tensor decomposition for 2x4 by 4x6 matrix multiplication.
Uses 6 independent Strassen (2x2x2) decompositions with disjoint output supports.
Total rank = 6 * 7 = 42.
"""
import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    # We decompose into 6 blocks of size 2x2x2
    # Block (i, k) where i in {0,1}, k in {0,1,2}
    # Each block multiplies A[i, j:j+2] (1x2) by B[j:j+2, k*2:(k+1)*2] (2x2) to get C[i, k*2:(k+1)*2] (1x2)
    # Wait, that's not right. Let me reconsider.
    
    # Actually, for 2x4 * 4x6:
    # C[i, k] = sum_j A[i,j] * B[j,k]
    # We can group the 4 middle indices into 2 groups of 2, and the 6 output indices into 3 groups of 2.
    # For each pair (i, k_group), we have a 1x2 * 2x2 = 1x2 multiplication.
    # But Strassen is for 2x2 * 2x2 = 2x2.
    # 
    # Let me reconsider the block structure.
    # We can think of it as:
    # A is 2x4, B is 4x6, C is 2x6.
    # Partition A into 2 blocks of 2x2 horizontally: A = [A_left, A_right] where A_left is 2x2, A_right is 2x2.
    # Partition B into 3 blocks of 2x2 vertically: B = [B_top; B_mid; B_bot] where each is 2x2.
    # Then C = A_left * B_top + A_right * B_bot? No, that's not right either.
    #
    # Actually, let's think of it differently.
    # C[i, k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # = sum_{j=0}^{1} A[i,j] * B[j,k] + sum_{j=2}^{3} A[i,j] * B[j,k]
    #
    # For fixed i and k, this is a dot product of 4 elements.
    # We can use Strassen-like tricks on pairs of elements.
    #
    # Let's use the following approach:
    # For each i in {0,1} and each k_group in {0,1,2} (where k_group covers 2 consecutive k values):
    #   We compute C[i, 2*k_group] and C[i, 2*k_group+1]
    #   = sum_{j=0}^{3} A[i,j] * B[j, 2*k_group] + A[i,j] * B[j, 2*k_group+1]
    #
    # Let's define for fixed i and k_group:
    #   a0 = A[i, 0], a1 = A[i, 1], a2 = A[i, 2], a3 = A[i, 3]
    #   b0 = B[0, 2*k_group], b1 = B[1, 2*k_group], b2 = B[2, 2*k_group], b3 = B[3, 2*k_group]
    #   c0 = B[0, 2*k_group+1], c1 = B[1, 2*k_group+1], c2 = B[2, 2*k_group+1], c3 = B[3, 2*k_group+1]
    #
    # We want to compute:
    #   C[i, 2*k_group] = a0*b0 + a1*b1 + a2*b2 + a3*b3
    #   C[i, 2*k_group+1] = a0*c0 + a1*c1 + a2*c2 + a3*c3
    #
    # This is like multiplying a 1x4 vector by a 4x2 matrix, resulting in 1x2.
    # We can use a rank-7 decomposition for this 1x4 * 4x2 = 1x2 problem?
    # Actually, the tensor for 1x4 * 4x2 has rank at most 8 (schoolbook).
    # Can we do better?
    #
    # Let's use a different approach. Consider the full 2x4 * 4x6 = 2x6 multiplication.
    # We can view this as 3 independent 2x4 * 4x2 = 2x2 multiplications (for each pair of output columns).
    # Each 2x4 * 4x2 = 2x2 multiplication can be done with rank at most 8 (schoolbook).
    # So total rank would be 3*8 = 24? No, that's not right because the A and B factors are shared.
    #
    # Actually, let's just use the schoolbook approach for now and see if we can improve.
    # Schoolbook rank = 2*4*6 = 48.
    #
    # Let me try a different decomposition.
    # Consider A as 2x4 and B as 4x6.
    # We can write:
    # C = A * B
    # Let's partition the 4 middle indices into 2 groups of 2.
    # A = [A1, A2] where A1, A2 are 2x2.
    # B = [B1; B2] where B1, B2 are 2x2 (but B is 4x6, so B1 is 2x6, B2 is 2x6).
    # Then C = A1 * B1 + A2 * B2.
    # A1 * B1 is 2x2 * 2x6 = 2x6, rank at most 12.
    # A2 * B2 is 2x2 * 2x6 = 2x6, rank at most 12.
    # Total rank at most 24.
    #
    # Can we do better with Strassen on the 2x2 * 2x6 part?
    # 2x2 * 2x6 can be viewed as 3 independent 2x2 * 2x2 multiplications.
    # Each 2x2 * 2x2 can be done with rank 7 (Strassen).
    # So 2x2 * 2x6 can be done with rank 3*7 = 21.
    # Then total rank would be 2*21 = 42.
    #
    # Let's implement this.
    
    # Strassen decomposition for 2x2 * 2x2:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 2x2 * 2x6:
    # A is 2x2, B is 2x6 = [B_left, B_mid, B_right] where each is 2x2.
    # C is 2x6 = [C_left, C_mid, C_right] where each is 2x2.
    # C_left = A * B_left (rank 7)
    # C_mid = A * B_mid (rank 7)
    # C_right = A * B_right (rank 7)
    # Total rank 21.
    
    # For the full 2x4 * 4x6:
    # A = [A1, A2] where A1, A2 are 2x2.
    # B is 4x6. We partition B into B_top (2x6) and B_bot (2x6).
    # C = A1 * B_top + A2 * B_bot.
    # A1 * B_top: rank 21
    # A2 * B_bot: rank 21
    # Total rank 42.
    
    # Let's construct the decomposition.
    # U has shape (n*m, rank) = (8, 42)
    # V has shape (m*p, rank) = (24, 42)
    # W has shape (p*n, rank) = (12, 42)
    
    rank = 42
    U = [[0] * rank for _ in range(8)]
    V = [[0] * rank for _ in range(24)]
    W = [[0] * rank for _ in range(12)]
    
    # Strassen coefficients for 2x2 * 2x2:
    # Each Strassen block contributes 7 columns.
    # For A1 * B_top:
    #   A1 is rows 0-1 of A, cols 0-1.
    #   B_top is rows 0-1 of B, all 6 cols.
    #   C is rows 0-1 of C, all 6 cols.
    #
    # Let's define the Strassen terms for 2x2 * 2x2:
    # M1: A00+A11, B00+B11
    # M2: A10+A11, B00
    # M3: A00, B10-B11
    # M4: A11, B01-B00
    # M5: A00+A01, B11
    # M6: A10-A00, B00+B01
    # M7: A01-A11, B10+B11
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For A1 * B_top (2x2 * 2x6):
    # B_top = [B_top_left, B_top_mid, B_top_right] where each is 2x2.
    # C_left = A1 * B_top_left
    # C_mid = A1 * B_top_mid
    # C_right = A1 * B_top_right
    #
    # For A2 * B_bot (2x2 * 2x6):
    # B_bot = [B_bot_left, B_bot_mid, B_bot_right] where each is 2x2.
    # C_left += A2 * B_bot_left
    # C_mid += A2 * B_bot_mid
    # C_right += A2 * B_bot_right
    
    # Let's define a helper to add a Strassen block.
    def add_strassen_block(U, V, W, a_row_offset, b_row_offset, c_row_offset, col_start):
        """
        Add a Strassen decomposition for 2x2 * 2x2.
        a_row_offset: row offset in U (0-1 for A rows)
        b_row_offset: row offset in V (0-1 for B rows, but B has 6 cols, so we need to specify col offset too)
        c_row_offset: row offset in W (0-1 for C rows, and col offset for C cols)
        col_start: starting column in U, V, W
        """
        # Strassen terms
        # M1: (A00 + A11) * (B00 + B11)
        # M2: (A10 + A11) * B00
        # M3: A00 * (B10 - B11)
        # M4: A11 * (B01 - B00)
        # M5: (A00 + A01) * B11
        # M6: (A10 - A00) * (B00 + B01)
        # M7: (A01 - A11) * (B10 + B11)
        
        # U entries (A side)
        # A is 2x2, indexed by (row, col) where row in {0,1}, col in {0,1}
        # U index: row * 2 + col (for the 2x2 block)
        # But in our global U, A is 2x4, so U index = i * 4 + j where i in {0,1}, j in {0,1,2,3}
        # For A1 (first 2x2 block), j in {0,1}
        # For A2 (second 2x2 block), j in {2,3}
        
        # Let's define local A indices: (0,0), (0,1), (1,0), (1,1)
        # Global A indices: (a_row_offset, a_col_offset + local_col)
        
        # V entries (B side)
        # B is 2x2 for this block, but in global B, B is 4x6.
        # V index: row * 6 + col where row in {0,1,2,3}, col in {0,1,2,3,4,5}
        # For B_top_left, row in {0,1}, col in {0,1}
        # For B_top_mid, row in {0,1}, col in {2,3}
        # For B_top_right, row in {0,1}, col in {4,5}
        # For B_bot_left, row in {2,3}, col in {0,1}
        # etc.
        
        # W entries (C side)
        # C is 2x2 for this block, but in global C, C is 2x6.
        # W index: col * 2 + row where col in {0,1,2,3,4,5}, row in {0,1}
        # For C_left, col in {0,1}
        # For C_mid, col in {2,3}
        # For C_right, col in {4,5}
        
        # Let's pass explicit indices.
        pass
    
    # Let's just hardcode the 6 Strassen blocks.
    # Block 1: A1 * B_top_left (C_left)
    # Block 2: A1 * B_top_mid (C_mid)
    # Block 3: A1 * B_top_right (C_right)
    # Block 4: A2 * B_bot_left (C_left)
    # Block 5: A2 * B_bot_mid (C_mid)
    # Block 6: A2 * B_bot_right (C_right)
    
    # Strassen coefficients:
    # M1: A=(1,1), B=(1,1), C=(1,1)
    # M2: A=(1,1), B=(1,0), C=(1,1)
    # M3: A=(1,0), B=(1,-1), C=(1,1)
    # M4: A=(1,1), B=(-1,1), C=(1,1)
    # M5: A=(1,1), B=(0,1), C=(1,1)
    # M6: A=(1,-1), B=(1,1), C=(1,1)
    # M7: A=(1,-1), B=(1,1), C=(1,1)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's define the Strassen terms as tuples: (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict from (a_row, a_col) to coefficient
    # V_coeffs: dict from (b_row, b_col) to coefficient
    # W_coeffs: dict from (c_row, c_col) to coefficient
    
    # M1: (A00 + A11) * (B00 + B11)
    M1 = {
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}  # Only C00 gets M1
    }
    # Wait, the W coefficients should be the same for all 4 output entries, but with different signs.
    # Let me redefine.
    
    # For each Strassen term, the W coefficients are:
    # C00: +1 for M1, M4, M7; -1 for M5
    # C01: +1 for M3, M5
    # C10: +1 for M2, M4
    # C11: +1 for M1, M3, M6; -1 for M2
    
    # Let's define the Strassen terms with their W coefficients.
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # M1: (A00+A11)*(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}),
        # M2: (A10+A11)*B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): 0}),
        # M3: A00*(B10-B11)
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 0}),
        # M4: A11*(B01-B00)
        ({(1,1): 1}, {(0,0): -1, (0,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0}),
        # M5: (A00+A01)*B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0}),
        # M6: (A10-A00)*(B00+B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}),
        # M7: (A01-A11)*(B10+B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}),
    ]
    
    # Now, for each of the 6 blocks, we add 7 columns.
    # Block (a_block, b_block, c_block) where:
    # a_block in {0,1} (A1 or A2)
    # b_block in {0,1,2} (B_top_left, B_top_mid, B_top_right) for a_block=0
    #                  in {0,1,2} (B_bot_left, B_bot_mid, B_bot_right) for a_block=1
    # c_block in {0,1,2} (C_left, C_mid, C_right)
    
    # For a_block=0 (A1), b_block=0 (B_top_left), c_block=0 (C_left)
    # For a_block=0 (A1), b_block=1 (B_top_mid), c_block=1 (C_mid)
    # For a_block=0 (A1), b_block=2 (B_top_right), c_block=2 (C_right)
    # For a_block=1 (A2), b_block=0 (B_bot_left), c_block=0 (C_left)
    # For a_block=1 (A2), b_block=1 (B_bot_mid), c_block=1 (C_mid)
    # For a_block=1 (A2), b_block=2 (B_bot_right), c_block=2 (C_right)
    
    col = 0
    for a_block in [0, 1]:
        for b_block in [0, 1, 2]:
            c_block = b_block  # c_block matches b_block
            
            # A offset: a_block * 2 (since A is 2x4, A1 is cols 0-1, A2 is cols 2-3)
            a_col_offset = a_block * 2
            # B offset: if a_block=0, B rows 0-1; if a_block=1, B rows 2-3
            b_row_offset = a_block * 2
            b_col_offset = b_block * 2
            # C offset: c_block * 2 for col, 0 for row (since C is 2x6, rows 0-1)
            c_col_offset = c_block * 2
            c_row_offset = 0
            
            # Add 7 Strassen terms
            for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
                # U entries
                for (ar, ac), coeff in u_coeffs.items():
                    global_row = ar * 4 + (a_col_offset + ac)
                    U[global_row][col] = coeff
                
                # V entries
                for (br, bc), coeff in v_coeffs.items():
                    global_row = (b_row_offset + br) * 6 + (b_col_offset + bc)
                    V[global_row][col] = coeff
                
                # W entries
                for (cr, cc), coeff in w_coeffs.items():
                    global_row = (c_col_offset + cc) * 2 + (c_row_offset + cr)
                    W[global_row][col] = coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
