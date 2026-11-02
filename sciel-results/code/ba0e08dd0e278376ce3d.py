"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-42 decomposition for 2x4 * 4x6 matrix multiplication
    using Strassen's algorithm on 2x2 blocks.
    
    The 2x4x6 multiplication tensor has 2*4*6 = 48 rank-1 terms in schoolbook.
    By viewing the 4 intermediate dimension as 2 blocks of size 2, and the 6 output
    dimension as 3 blocks of size 2, we can apply Strassen to each 2x2x2 subproblem.
    
    Strassen reduces 8 scalar mults to 7 for 2x2x2.
    We have 2 (i-blocks) * 3 (k-blocks) = 6 such subproblems... 
    Actually, let's think more carefully.
    
    A is 2x4, B is 4x6. Let's partition:
    A = [A0 | A1] where A0, A1 are 2x2
    B = [B0; B1; B2] where B0, B1, B2 are 2x2 (stacked vertically, each 2 rows x 2 cols for a column block)
    
    Wait, B is 4x6. Let's partition B into 3 blocks of 2x2 columns:
    B = [B_{j=0} | B_{j=1} | B_{j=2}] where each B_j is 4x2.
    Then further split B_j into top and bottom 2x2:
    B_j = [B_{j,0}; B_{j,1}] where each is 2x2.
    
    Similarly A = [A_0 | A_1] where A_0, A_1 are 2x2.
    
    C = A * B, C is 2x6.
    C's column j (2x2) = A_0 * B_{j,0} + A_1 * B_{j,1}
    
    For each j, we need to compute C_j = A_0 * B_{j,0} + A_1 * B_{j,1}
    
    We can use Strassen-like tricks. Let's define for each j:
    The product A_0 * B_{j,0} and A_1 * B_{j,1} are both 2x2.
    
    Actually, let's just directly construct the rank-1 terms.
    
    Let me reconsider. The tensor T(i, j, k) = delta for C[i,k] = sum_j A[i,j] B[j,k].
    
    Schoolbook: rank = 2*4*6 = 48.
    
    Using Strassen on 2x2 blocks: We partition the j dimension (size 4) into two halves of size 2.
    A = [A_left | A_right] (each 2x2)
    B = [B_top; B_bottom] where B_top and B_bottom are 2x6.
    Further split B_top into 3 blocks of 2x2, and B_bottom into 3 blocks of 2x2.
    
    C = A_left * B_top + A_right * B_bottom
    
    For each of the 3 column-pairs in B:
    C[:, 2j:2j+2] = A_left * B_top[:, 2j:2j+2] + A_right * B_bottom[:, 2j:2j+2]
    
    Each of these is a sum of two 2x2 matrix multiplications.
    
    Standard Strassen for 2x2: if we want to compute P = X*Y + Z*W where all are 2x2,
    we can't directly apply Strassen. But we can think of it differently.
    
    Let me just enumerate the rank-1 terms directly using the Strassen idea.
    
    For a 2x2 * 2x2 multiplication, Strassen uses 7 terms instead of 8.
    
    Let's apply this per column-block. For column block j (j=0,1,2):
    We need C[:, 2j:2j+2] = A_L * B_T[:,2j:2j+2] + A_R * B_B[:,2j:2j+2]
    
    where A_L, A_R are 2x2, and B_T[:,2j:2j+2], B_B[:,2j:2j+2] are 2x2.
    
    Let X = A_L, Y = B_T[:,2j:2j+2], Z = A_R, W = B_B[:,2j:2j+2].
    We want X*Y + Z*W.
    
    This is equivalent to multiplying a 2x4 matrix [X|Z] by a 4x2 matrix [Y;W].
    
    We can use a rank-7 decomposition for 2x2x2, but here we have 2x4x2 which is 8 terms in schoolbook.
    Can we do it in 7? 
    
    Actually, the minimum rank for 2x4x2 is not 7. Let me think...
    
    The bilinear complexity of 2x4x2: This is the same as 2x2x2 * 2 + some overlap. 
    Actually, 2x4x2 has rank at least 8 by standard arguments? No, let me reconsider.
    
    Let me just use 8 terms per column block, giving 3*8 = 24... no that's not right either.
    
    Let me just directly write out the schoolbook and then apply Strassen to each 2x2x2 subproblem.
    
    Actually, the cleanest approach: 
    n=2, m=4, p=6.
    Split m into two parts of size 2: m1=2, m2=2.
    Split p into three parts of size 2: p1=2, p2=2, p3=2.
    
    A = [A0 | A1], A0 is 2x2, A1 is 2x2.
    B has 4 rows, 6 cols. Split rows: B = [B_top; B_bot], each 2x6. Split cols into 3 groups of 2.
    
    For each column group c (0,1,2):
    C[:, 2c:2c+2] = A0 * B_top[:, 2c:2c+2] + A1 * B_bot[:, 2c:2c+2]
    
    Let's denote B_top_c = B_top[:, 2c:2c+2] (2x2), B_bot_c = B_bot[:, 2c:2c+2] (2x2).
    
    C_c = A0 * B_top_c + A1 * B_bot_c
    
    Now, A0*B_top_c is a 2x2 matmul (8 terms in schoolbook, 7 in Strassen).
    A1*B_bot_c is a 2x2 matmul (8 terms, 7 in Strassen).
    
    But we're summing two 2x2 matmuls. The total is 16 terms in schoolbook for C_c.
    With Strassen on each: 7+7 = 14 terms.
    
    For 3 column groups: 3*14 = 42 terms. That matches!
    
    So the construction is: for each column group c, apply Strassen to A0*B_top_c and A1*B_bot_c separately,
    giving 7+7 = 14 rank-1 terms per column group, total 42.
    """
    deadline = time.time() + 140  # 140s safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Strassen for 2x2: 
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For a 2x2 matmul X*Y, the rank-1 terms are:
    # Term for C00: 
    #   M1: X00+X11 times Y00+Y11 -> contributes to C00 with coeff +1
    #   M4: X11*(Y01-Y10) -> contributes to C00 with coeff +1
    #   M5: (X00+X01)*Y11 -> contributes to C00 with coeff -1
    #   M7: (X01-X11)*(Y10+Y11) -> contributes to C00 with coeff +1
    # ... etc.
    
    # Let's build the factor matrices.
    # U has shape (n*m, rank) = (8, 42)
    # V has shape (m*p, rank) = (24, 42)  
    # W has shape (p*n, rank) = (12, 42)
    
    # Indexing:
    # U[i*m + j, col] - A element at row i, col j
    # V[j*p + k, col] - B element at row j, col k
    # W[k*n + i, col] - C element at row i, col k (trace convention)
    
    U = [[0.0] * 42 for _ in range(8)]
    V = [[0.0] * 42 for _ in range(24)]
    W = [[0.0] * 42 for _ in range(12)]
    
    col = 0
    
    # Strassen definitions for 2x2 matmul X*Y:
    # Each "M" term is a product of a linear form in X and a linear form in Y.
    # We represent each rank-1 term by:
    #   u_coeffs: dict from (i, j) to coeff, for X elements
    #   v_coeffs: dict from (j2, k) to coeff, for Y elements  
    #   w_coeffs: dict from (k2, i2) to coeff, for C elements
    
    # Strassen M terms:
    # M1: X = (X00+X11), Y = (Y00+Y11)
    # M2: X = (X10+X11), Y = Y00
    # M3: X = X00, Y = (Y10-Y11)
    # M4: X = X11, Y = (Y01-Y10)
    # M5: X = (X00+X01), Y = Y11
    # M6: X = (X10-X00), Y = (Y00+Y01)
    # M7: X = (X01-X11), Y = (Y10+Y11)
    
    # Contributions to C:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_terms = [
        # (x_coeffs, y_coeffs, c_contribs)
        # x_coeffs: {(0,0): 1, (1,1): 1} means X00 + X11
        # y_coeffs: {(0,0): 1, (1,1): 1} means Y00 + Y11
        # c_contribs: {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0} etc.
        
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, 
         {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, 
         {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): -1}),
        # M3
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}, 
         {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 1}),
        # M4
        ({(1,1): 1}, {(0,1): 1, (1,0): -1}, 
         {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0}),
        # M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, 
         {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0}),
        # M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, 
         {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}),
        # M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, 
         {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}),
    ]
    
    for c_group in range(3):  # 3 column groups
        # For A0 * B_top_c:
        # A0 is rows 0-1, cols 0-1 of A (2x4)
        # B_top_c is rows 0-1, cols 2*c_group to 2*c_group+1 of B (4x6)
        
        for (x_coeffs, y_coeffs, c_contribs) in strassen_terms:
            if col >= 42:
                break
            for (xi, xj), xc in x_coeffs.items():
                # A element at row xi, col xj
                U[xi * 4 + xj][col] += xc
            for (yi, yj), yc in y_coeffs.items():
                # B element at row yi, col 2*c_group + yj
                V[yi * 6 + (2*c_group + yj)][col] += yc
            for (ki, ki2), wc in c_contribs.items():
                # C element at row ki, col 2*c_group + ki2
                if wc != 0:
                    W[(2*c_group + ki2) * 2 + ki][col] += wc
            col += 1
        
        # For A1 * B_bot_c:
        # A1 is rows 0-1, cols 2-3 of A (2x4)
        # B_bot_c is rows 2-3, cols 2*c_group to 2*c_group+1 of B (4x6)
        
        for (x_coeffs, y_coeffs, c_contribs) in strassen_terms:
            if col >= 42:
                break
            for (xi, xj), xc in x_coeffs.items():
                # A element at row xi, col 2 + xj
                U[xi * 4 + (2 + xj)][col] += xc
            for (yi, yj), yc in y_coeffs.items():
                # B element at row 2 + yi, col 2*c_group + yj
                V[(2 + yi) * 6 + (2*c_group + yj)][col] += yc
            for (ki, ki2), wc in c_contribs.items():
                # C element at row ki, col 2*c_group + ki2
                if wc != 0:
                    W[(2*c_group + ki2) * 2 + ki][col] += wc
            col += 1
    
    if time.time() > deadline:
        # Shouldn't happen, but return what we have
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
