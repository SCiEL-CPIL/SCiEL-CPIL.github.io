import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    
    # Rank 28 decomposition based on 4 Strassen 2x2x2 blocks
    rank = 28
    
    # Initialize U, V, W with zeros
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen coefficients for 2x2x2 multiplication
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Local indices for 2x2 matrices:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    strassen_coeffs = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U coeffs map local A indices to coefficients
        # V coeffs map local B indices to coefficients
        # W coeffs map local C indices to coefficients
        
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def add_block(a_row_off, b_row_off, b_col_off, c_col_off, start_col):
        """
        Adds a Strassen block to U, V, W.
        a_row_off: row offset in A (0 or 2)
        b_row_off: row offset in B (0 or 2)
        b_col_off: col offset in B (0 or 2)
        c_col_off: col offset in C (0 or 2)
        start_col: starting column index in U, V, W (0, 7, 14, 21)
        """
        for s in range(7):
            col = start_col + s
            u_coeffs, v_coeffs, w_coeffs = strassen_coeffs[s]
            
            # Update U
            for local_idx, coeff in u_coeffs.items():
                r = local_idx // 2
                c = local_idx % 2
                global_r = a_row_off + r
                global_c = c
                # A is n x m. Row index in U is global_r * m + global_c
                # Wait, A is 2x4.
                # Block A0 is rows 0-1, cols 0-1.
                # Block A1 is rows 0-1, cols 2-3?
                # No, A is 2x4.
                # We split A into two 2x2 blocks horizontally: A_left (cols 0,1), A_right (cols 2,3).
                # So a_row_off should be 0 for both, but a_col_off should be 0 or 2.
                # Let's re-verify the partitioning used in the original code.
                
                # Original code:
                # emit_block(0, 0, 0, 0, 0)  -> A0, B00 -> C0
                # emit_block(0, 0, 2, 2, 7)  -> A0, B01 -> C1
                # emit_block(2, 2, 0, 0, 14) -> A1, B10 -> C0
                # emit_block(2, 2, 2, 2, 21) -> A1, B11 -> C1
                
                # This implies A is split vertically?
                # A is 2x4.
                # If A is split into A_top (rows 0) and A_bottom (rows 1)? No, 2x2 blocks.
                # A must be split into two 2x2 blocks.
                # Option 1: Horizontal split. A_left (cols 0,1), A_right (cols 2,3).
                # Then A_left is 2x2, A_right is 2x2.
                # B is 4x4. Split into 2x2 blocks B00, B01, B10, B11.
                # C is 2x4. Split into C_left (cols 0,1), C_right (cols 2,3).
                
                # C_left = A_left B00 + A_right B10
                # C_right = A_left B01 + A_right B11
                
                # In this case:
                # Block 1: A_left (cols 0,1), B00 (rows 0,1; cols 0,1) -> C_left (cols 0,1)
                # Block 2: A_left (cols 0,1), B01 (rows 0,1; cols 2,3) -> C_right (cols 2,3)
                # Block 3: A_right (cols 2,3), B10 (rows 2,3; cols 0,1) -> C_left (cols 0,1)
                # Block 4: A_right (cols 2,3), B11 (rows 2,3; cols 2,3) -> C_right (cols 2,3)
                
                # So for A, the "row offset" in the 2x2 local sense is always 0 (since A has only 2 rows).
                # The "col offset" in A is 0 for A_left, 2 for A_right.
                
                # The original code used `a_cs` as the offset in A's column space?
                # Let's look at `U[r * m + a_cs + cc][col]`.
                # `r` is 0 or 1 (local row).
                # `cc` is 0 or 1 (local col).
                # `a_cs` is the global column offset in A.
                # This matches the horizontal split of A.
                
                U[global_r * m + (a_col_off + c)][col] += coeff
            
            # Update V
            for local_idx, coeff in v_coeffs.items():
                r = local_idx // 2
                c = local_idx % 2
                global_r = b_row_off + r
                global_c = b_col_off + c
                # B is 4x4. Index in V is global_r * p + global_c
                V[global_r * p + global_c][col] += coeff
            
            # Update W
            for local_idx, coeff in w_coeffs.items():
                r = local_idx // 2
                c = local_idx % 2
                # C is 2x4.
                # C_left has cols 0,1. C_right has cols 2,3.
                # The local C index `c` is 0 or 1.
                # The global column offset `c_col_off` is 0 or 2.
                # Global row is `r` (0 or 1).
                # W index: `global_r * n + global_c`?
                # W shape is (p*n, rank) = (8, 28).
                # The task says "output factor follows the official trace convention W[k*n+i,r]".
                # This usually means W is indexed by output index and rank.
                # Output C is 2x4. Total elements 8.
                # Index in C: row `r`, col `c_global`.
                # Linear index: `r * 4 + c_global`? Or `c_global * 2 + r`?
                # "W[k*n+i,r]" suggests k is column, i is row?
                # Standard row-major: index = row * cols + col.
                # Here n=2 rows, p=4 cols? No, C is n x p = 2 x 4.
                # So rows=2, cols=4.
                # Index = row * 4 + col.
                
                # Let's check the original code's W indexing:
                # W[(c_cs + cc) * n + r][col]
                # c_cs is col offset (0 or 2). cc is local col (0 or 1).
                # So global col = c_cs + cc.
                # r is local row (0 or 1).
                # Index = (global_col) * n + r.
                # This is column-major indexing for C?
                # If C is 2x4, column-major index for (r, c) is c * 2 + r.
                # Yes, this matches.
                
                global_c = c_col_off + c
                W[global_c * n + r][col] += coeff

    # Block 1: A_left, B00 -> C_left
    # A_col_off=0, B_row_off=0, B_col_off=0, C_col_off=0
    add_block(0, 0, 0, 0, 0)
    
    # Block 2: A_left, B01 -> C_right
    # A_col_off=0, B_row_off=0, B_col_off=2, C_col_off=2
    add_block(0, 0, 2, 2, 7)
    
    # Block 3: A_right, B10 -> C_left
    # A_col_off=2, B_row_off=2, B_col_off=0, C_col_off=0
    add_block(2, 2, 0, 0, 14)
    
    # Block 4: A_right, B11 -> C_right
    # A_col_off=2, B_row_off=2, B_col_off=2, C_col_off=2
    add_block(2, 2, 2, 2, 21)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
