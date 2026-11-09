"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=6
    
    # We will build the decomposition in parts.
    # Part 1: Bottom row of A (1x4) * B (4x6) -> Rank 24
    # Part 2: Top 2 rows of A (2x4) * B (4x6) -> Rank 42
    
    rank = 66
    U = [[0] * rank for _ in range(n * m)] # 12 x 66
    V = [[0] * rank for _ in range(m * p)] # 24 x 66
    W = [[0] * rank for _ in range(p * n)] # 18 x 66
    
    col = 0
    
    # --- Part 1: Bottom row (i=2) ---
    # A[2, j] * B[j, k]
    # U index: i*m + j = 2*4 + j = 8+j
    # V index: j*p + k
    # W index: k*n + i = k*3 + 2
    for j in range(4):
        for k in range(6):
            U[8 + j][col] = 1
            V[j * 6 + k][col] = 1
            W[k * 3 + 2][col] = 1
            col += 1
            
    # --- Part 2: Top 2 rows (i=0,1) ---
    # A_top is 2x4. B is 4x6.
    # We split B into B_4 (4x4, cols 0-3) and B_2 (4x2, cols 4-5).
    # C_top = A_top B_4 + A_top B_2
    
    # --- Part 2a: A_top (2x4) * B_4 (4x4) ---
    # Split A_top into A1 (2x2, cols 0-1) and A2 (2x2, cols 2-3).
    # Split B_4 into B41 (2x4, rows 0-1) and B42 (2x4, rows 2-3).
    # C = A1 B41 + A2 B42
    # A1 B41 is 2x2x4. Rank 14.
    # A2 B42 is 2x2x4. Rank 14.
    
    def add_2x2x4(A_row_start, B_row_start, B_col_start, C_row_start, C_col_start, current_col):
        # A is 2x2 at rows A_row_start..+1, cols 0..1 relative to A_block?
        # Actually, let's map indices explicitly.
        # A1: rows 0,1 of A_top (global 0,1). Cols 0,1 of A (global 0,1).
        # B41: rows 0,1 of B (global 0,1). Cols 0,3 of B_4 (global 0,3).
        
        # We need to decompose 2x2x4.
        # Split the 4 columns of B into two 2x2 blocks.
        # B_block1: 2x2 (cols 0,1)
        # B_block2: 2x2 (cols 2,3)
        
        # Term 1: A (2x2) * B1 (2x2) -> 2x2. Rank 7.
        # Term 2: A (2x2) * B2 (2x2) -> 2x2. Rank 7.
        
        # Helper to add 2x2x2 Strassen
        def add_2x2x2(a_r1, a_r2, a_c1, a_c2, b_r1, b_r2, b_c1, b_c2, c_r1, c_r2, c_c1, c_c2, c_col):
            # a_r1, a_r2 are global row indices for A (0,1)
            # a_c1, a_c2 are global col indices for A (0,1)
            # b_r1, b_r2 are global row indices for B (0,1)
            # b_c1, b_c2 are global col indices for B (0,1)
            # c_r1, c_r2 are global row indices for C (0,1)
            # c_c1, c_c2 are global col indices for C (0,1)
            
            # U indices: i*m + j
            # V indices: j*p + k
            # W indices: k*n + i
            
            # M1 = (A11 + A22)(B11 + B22)
            # A11: a_r1, a_c1. A22: a_r2, a_c2.
            # B11: b_r1, b_c1. B22: b_r2, b_c2.
            
            # We add 7 columns.
            # 1. M1
            U[a_r1*m + a_c1][c_col] += 1
            U[a_r2*m + a_c2][c_col] += 1
            V[b_r1*p + b_c1][c_col] += 1
            V[b_r2*p + b_c2][c_col] += 1
            W[b_c1*n + a_r1][c_col] += 1
            W[b_c1*n + a_r2][c_col] += 1
            W[b_c2*n + a_r1][c_col] += 1
            W[b_c2*n + a_r2][c_col] += 1
            c_col += 1
            
            # 2. M2 = (A21 + A22)B11
            # A21: a_r2, a_c1. A22: a_r2, a_c2.
            # B11: b_r1, b_c1.
            # Result contributes to C21, C22?
            # C21 = M2 + M4. C22 = M1 - M2 + ...
            # So M2 contributes to C(a_r2, b_c1) with +1 and C(a_r2, b_c2) with -1.
            U[a_r2*m + a_c1][c_col] += 1
            U[a_r2*m + a_c2][c_col] += 1
            V[b_r1*p + b_c1][c_col] += 1
            W[b_c1*n + a_r2][c_col] += 1
            W[b_c2*n + a_r2][c_col] -= 1
            c_col += 1
            
            # 3. M3 = A11(B12 - B22)
            # A11: a_r1, a_c1.
            # B12: b_r1, b_c2. B22: b_r2, b_c2.
            # Result contributes to C12, C22.
            # C12 = M3 + M5. C22 = ... + M3.
            U[a_r1*m + a_c1][c_col] += 1
            V[b_r1*p + b_c2][c_col] += 1
            V[b_r2*p + b_c2][c_col] -= 1
            W[b_c2*n + a_r1][c_col] += 1
            W[b_c2*n + a_r2][c_col] += 1
            c_col += 1
            
            # 4. M4 = A22(B21 - B11)
            # A22: a_r2, a_c2.
            # B21: b_r2, b_c1. B11: b_r1, b_c1.
            # Result contributes to C11, C21.
            # C11 = M1 + M4 ... C21 = M2 + M4.
            U[a_r2*m + a_c2][c_col] += 1
            V[b_r2*p + b_c1][c_col] += 1
            V[b_r1*p + b_c1][c_col] -= 1
            W[b_c1*n + a_r1][c_col] += 1
            W[b_c1*n + a_r2][c_col] += 1
            c_col += 1
            
            # 5. M5 = (A11 + A12)B22
            # A11: a_r1, a_c1. A12: a_r1, a_c2.
            # B22: b_r2, b_c2.
            # Result contributes to C11, C12.
            # C11 = ... - M5. C12 = M3 + M5.
            U[a_r1*m + a_c1][c_col] += 1
            U[a_r1*m + a_c2][c_col] += 1
            V[b_r2*p + b_c2][c_col] += 1
            W[b_c2*n + a_r1][c_col] -= 1
            W[b_c2*n + a_r2][c_col] += 1 # Wait, C12 is row 0?
            # C is 2x2. Rows 0,1. Cols 0,1.
            # C11 is (0,0). C12 is (0,1). C21 is (1,0). C22 is (1,1).
            # M5 = (A11+A12)B22.
            # A11+A12 is row 0 of A. B22 is col 1 of B? No, B22 is block (1,1).
            # Let's re-verify Strassen indices.
            # A = [[A11, A12], [A21, A22]]
            # B = [[B11, B12], [B21, B22]]
            # C11 = A11 B11 + A12 B21
            # C12 = A11 B12 + A12 B22
            # C21 = A21 B11 + A22 B21
            # C22 = A21 B12 + A22 B22
            
            # M5 = (A11 + A12) B22.
            # (A11 + A12) is Row 0 of A.
            # B22 is Block (1,1) of B.
            # Product is Row 0 of C? No.
            # (Row 0 of A) * (Block (1,1) of B).
            # This contributes to C11? No.
            # C11 uses A11, A12.
            # Let's check standard Strassen:
            # M5 = (A11 + A12) B22.
            # C11 = M1 + M4 - M5 + M7.
            # C12 = M3 + M5.
            # So M5 contributes to C11 (-1) and C12 (+1).
            # C11 is (0,0). C12 is (0,1).
            # So W indices:
            # C11: k=b_c1 (col 0 of B block), i=a_r1 (row 0 of A block).
            # C12: k=b_c2 (col 1 of B block), i=a_r1 (row 0 of A block).
            
            W[b_c1*n + a_r1][c_col] -= 1
            W[b_c2*n + a_r1][c_col] += 1
            c_col += 1
            
            # 6. M6 = (A21 - A11)(B11 + B12)
            # A21: a_r2, a_c1. A11: a_r1, a_c1.
            # B11: b_r1, b_c1. B12: b_r1, b_c2.
            # Result contributes to C22.
            # C22 = M1 - M2 + M3 + M6.
            # So M6 contributes to C22 (+1).
            # C22 is (1,1). k=b_c2? No, B11+B12 is Row 0 of B block.
            # Wait. B11 is (0,0), B12 is (0,1). Sum is Row 0.
            # (A21 - A11) is Col 0 of A block? No.
            # A21 is (1,0), A11 is (0,0). Difference is Col 0.
            # Product: (Col 0 of A) * (Row 0 of B).
            # This is a scalar? No, A is 2x2, B is 2x2.
            # A21-A11 is a column vector? No, A is matrix.
            # A21 is element (1,0). A11 is element (0,0).
            # (A21 - A11) is not a matrix.
            # Strassen uses block matrices.
            # If we are doing 2x2x2, the "blocks" are scalars.
            # So A11, A12 etc are scalars.
            # M6 = (A21 - A11) * (B11 + B12).
            # This is a scalar multiplication.
            # It contributes to C22.
            # C22 is element (1,1).
            # So W index: k=b_c2 (col 1), i=a_r2 (row 1).
            
            U[a_r2*m + a_c1][c_col] -= 1 # A21 - A11
            U[a_r1*m + a_c1][c_col] += 1 # Wait, (A21 - A11) means coeff of A21 is 1, A11 is -1.
            # My U update: U[row][col] += coeff.
            # A21 is at a_r2, a_c1. Coeff 1.
            # A11 is at a_r1, a_c1. Coeff -1.
            U[a_r2*m + a_c1][c_col] = 1 # Reset? No, I should just add.
            # I am building from scratch, so I can just set.
            # But I am reusing the function.
            # Let's assume U,V,W are zeroed.
            
            U[a_r2*m + a_c1][c_col] += 1
            U[a_r1*m + a_c1][c_col] -= 1
            V[b_r1*p + b_c1][c_col] += 1 # B11
            V[b_r1*p + b_c2][c_col] += 1 # B12
            W[b_c2*n + a_r2][c_col] += 1
            c_col += 1
            
            # 7. M7 = (A12 - A22)(B21 + B22)
            # A12: a_r1, a_c2. A22: a_r2, a_c2.
            # B21: b_r2, b_c1. B22: b_r2, b_c2.
            # Result contributes to C11.
            # C11 = M1 + M4 - M5 + M7.
            # M7 contributes to C11 (+1).
            # C11 is (0,0).
            
            U[a_r1*m + a_c2][c_col] += 1
            U[a_r2*m + a_c2][c_col] -= 1
            V[b_r2*p + b_c1][c_col] += 1
            V[b_r2*p + b_c2][c_col] += 1
            W[b_c1*n + a_r1][c_col] += 1
            c_col += 1
            
            return c_col

        # Call for B1 (cols 0,1)
        # A1 is rows 0,1 cols 0,1.
        # B1 is rows 0,1 cols 0,1.
        # C1 is rows 0,1 cols 0,1.
        c_col = add_2x2x2(0, 1, 0, 1, B_row_start, B_row_start+1, B_col_start, B_col_start+1, C_row_start, C_row_start+1, C_col_start, C_col_start+1, c_col)
        
        # Call for B2 (cols 2,3)
        # A1 is rows 0,1 cols 0,1.
        # B2 is rows 0,1 cols 2,3.
        # C2 is rows 0,1 cols 2,3.
        c_col = add_2x2x2(0, 1, 0, 1, B_row_start, B_row_start+1, B_col_start+2, B_col_start+3, C_row_start, C_row_start+1, C_col_start+2, C_col_start+3, c_col)
        
        return c_col

    # A1 (2x2) * B41 (2x4)
    # A1 rows 0,1. Cols 0,1.
    # B41 rows 0,1. Cols 0,3.
    # C11 rows 0,1. Cols 0,3.
    col = add_2x2x4(0, 0, 0, 0, 0, col)
    
    # A2 (2x2) * B42 (2x4)
    # A2 rows 0,1. Cols 2,3.
    # B42 rows 2,3. Cols 0,3.
    # C12 rows 0,1. Cols 0,3.
    # Note: A2 is at cols 2,3. So a_c1=2, a_c2=3.
    # B42 is at rows 2,3. So b_r1=2, b_r2=3.
    # C is at cols 0,3.
    col = add_2x2x4(0, 2, 0, 2, 2, col) # Wait, add_2x2x4 signature needs adjustment or I pass offsets.
    
    # My add_2x2x4 was hardcoded for A1/B41.
    # I need to generalize it or write specific calls.
    
    # Let's rewrite the logic to be more flexible.
    # Instead of a generic function, I'll just call add_2x2x2 4 times for the 2x4 part.
    
    # --- Redo Part 2a with explicit calls ---
    # A1 (2x2) * B41 (2x4) -> Two 2x2x2
    # 1. A1 * B41_left (2x2)
    # A1: r 0,1 c 0,1. B41_l: r 0,1 c 0,1. C: r 0,1 c 0,1.
    col = add_2x2x2(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, col)
    # 2. A1 * B41_right (2x2)
    # A1: r 0,1 c 0,1. B41_r: r 0,1 c 2,3. C: r 0,1 c 2,3.
    col = add_2x2x2(0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3, col)
    
    # A2 (2x2) * B42 (2x4) -> Two 2x2x2
    # 3. A2 * B42_left (2x2)
    # A2: r 0,1 c 2,3. B42_l: r 2,3 c 0,1. C: r 0,1 c 0,1.
    col = add_2x2x2(0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 0, 1, col)
    # 4. A2 * B42_right (2x2)
    # A2: r 0,1 c 2,3. B42_r: r 2,3 c 2,3. C: r 0,1 c 2,3.
    col = add_2x2x2(0, 1, 2, 3, 2, 3, 2, 3, 0, 1, 2, 3, col)
    
    # --- Part 2b: A_top (2x4) * B_2 (4x2) ---
    # B_2 is cols 4,5.
    # Split A_top into A1 (2x2, c 0,1) and A2 (2x2, c 2,3).
    # Split B_2 into B21 (2x2, r 0,1 c 4,5) and B22 (2x2, r 2,3 c 4,5).
    # C = A1 B21 + A2 B22.
    
    # 5. A1 * B21
    # A1: r 0,1 c 0,1. B21: r 0,1 c 4,5. C: r 0,1 c 4,5.
    col = add_2x2x2(0, 1, 0, 1, 0, 1, 4, 5, 0, 1, 4, 5, col)
    
    # 6. A2 * B22
    # A2: r 0,1 c 2,3. B22: r 2,3 c 4,5. C: r 0,1 c 4,5.
    col = add_2x2x2(0, 1, 2, 3, 2, 3, 4, 5, 0, 1, 4, 5, col)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
