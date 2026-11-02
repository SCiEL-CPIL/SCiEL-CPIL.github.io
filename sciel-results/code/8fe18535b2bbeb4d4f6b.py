"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=8
    # Rank 88 construction
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 decomposition for a specific block
    # A_block: (start_row, start_col, 2, 2) in A
    # B_block: (start_row, start_col, 2, 2) in B
    # C_block: (start_row, start_col, 2, 2) in C
    # Note: A is 3x4, B is 4x8, C is 8x3 (W index k*n+i)
    
    def add_strassen_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        # Strassen for 2x2 matrices
        # P1 = A11 * B11
        # P2 = (A11 + A21) * B12
        # P3 = A21 * (B12 - B22)
        # P4 = A22 * B22
        # P5 = (A11 + A22) * (B11 + B22)
        # P6 = (A12 - A22) * (B21 + B22)
        # P7 = (A11 - A21) * (B11 + B12)
        
        # Coefficients for U, V, W
        # U indices: a_r + dr, a_c + dc
        # V indices: b_r + dr, b_c + dc
        # W indices: c_r + dr, c_c + dc -> W[(c_r+dr)*n + (c_c+dc)]
        
        # P1
        U[a_r][col] += 1
        U[a_r+1][col] += 0 # A11 is (0,0)
        # Wait, A11 is 2x2.
        # P1 = A11 * B11.
        # This is not a single scalar multiplication.
        # It is a 2x2 matrix multiplication.
        # My previous logic was that we replace the 2x2 matrix multiplication with 7 scalar multiplications.
        # So for each "scalar" multiplication in Strassen, we need to add rank-1 tensors for the actual scalar multiplications.
        
        # Actually, the Strassen algorithm computes 7 products of 2x2 matrices?
        # No, Strassen computes 7 products of "scalars".
        # If the "scalars" are 2x2 matrices, then we need to decompose each 2x2 matrix product.
        
        # Let's re-read the rank calculation.
        # Rank of 2x2 matrix multiplication is 7.
        # So we can just call a helper that adds a rank-7 decomposition for a 2x2 matrix multiplication.
        
        pass # Placeholder

    # This approach is getting complicated to code inline.
    # Let's use the explicit 88 rank construction by iterating over the 4 blocks of 2x2 matrix multiplications.
    
    # We will define a helper to add a rank-7 decomposition for a 2x2 matrix multiplication
    # given the start indices of A, B, C blocks.
    
    def add_rank7_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        # Strassen products for 2x2 matrix multiplication
        # A = [[A11, A12], [A21, A22]]
        # B = [[B11, B12], [B21, B22]]
        
        # P1 = A11 * B11
        # P2 = (A11 + A21) * B12
        # P3 = A21 * (B12 - B22)
        # P4 = A22 * B22
        # P5 = (A11 + A22) * (B11 + B22)
        # P6 = (A12 - A22) * (B21 + B22)
        # P7 = (A11 - A21) * (B11 + B12)
        
        # Each Pi is a scalar multiplication.
        # But A11 etc are scalars in the 2x2 matrix context.
        # So we just add 1 to U, V, W at the appropriate scalar positions.
        
        # U index: (a_r + dr) * m + (a_c + dc)
        # V index: (b_r + dr) * p + (b_c + dc)
        # W index: (c_r + dr) * n + (c_c + dc)
        
        def add_scalar_u(dr_u, dc_u, val):
            U[a_r + dr_u][col] += val * (1 if a_c + dc_u < m else 0) # Bounds check not needed if correct
        def add_scalar_v(dr_v, dc_v, val):
            V[b_r + dr_v][col] += val
        def add_scalar_w(dr_w, dc_w, val):
            W[(c_r + dr_w) * n + (c_c + dc_w)][col] += val
            
        # P1
        add_scalar_u(0, 0, 1)
        add_scalar_v(0, 0, 1)
        add_scalar_w(0, 0, 1)
        col += 1
        
        # P2
        add_scalar_u(0, 0, 1)
        add_scalar_u(1, 0, 1)
        add_scalar_v(0, 1, 1)
        add_scalar_w(0, 1, 1)
        col += 1
        
        # P3
        add_scalar_u(1, 0, 1)
        add_scalar_v(0, 1, 1)
        add_scalar_v(1, 1, -1)
        add_scalar_w(1, 1, 1)
        col += 1
        
        # P4
        add_scalar_u(1, 1, 1)
        add_scalar_v(1, 1, 1)
        add_scalar_w(1, 1, 1)
        col += 1
        
        # P5
        add_scalar_u(0, 0, 1)
        add_scalar_u(1, 1, 1)
        add_scalar_v(0, 0, 1)
        add_scalar_v(1, 1, 1)
        add_scalar_w(0, 0, 1)
        add_scalar_w(1, 1, -1)
        col += 1
        
        # P6
        add_scalar_u(0, 1, 1)
        add_scalar_u(1, 1, -1)
        add_scalar_v(1, 0, 1)
        add_scalar_v(1, 1, 1)
        add_scalar_w(1, 0, 1)
        col += 1
        
        # P7
        add_scalar_u(0, 0, 1)
        add_scalar_u(1, 0, -1)
        add_scalar_v(0, 0, 1)
        add_scalar_v(0, 1, 1)
        add_scalar_w(0, 0, -1)
        col += 1

    # Block 1: A_bot (row 2) * B_1 (cols 0-3)
    # A_bot is 1x4. B_1 is 4x4.
    # This is 16 scalar multiplications.
    for j in range(4):
        for k in range(4):
            U[2][col] = 1
            V[k * 8 + j][col] = 1
            W[(j) * 3 + 2][col] = 1
            col += 1
            
    # Block 2: A_top (rows 0-1) * B_1 (cols 0-3)
    # A_top is 2x4. B_1 is 4x4.
    # Split into 4 2x2 matrix multiplications.
    # A_top = [[A11, A12]]
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # B_1 = [[B11, B12], [B21, B22]]
    # B11: rows 0-1, cols 0-1
    # B12: rows 0-1, cols 2-3
    # B21: rows 2-3, cols 0-1
    # B22: rows 2-3, cols 2-3
    
    # C_top = [[C11, C12]]
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    
    # We need to add the rank-7 decompositions for these 4 products.
    # Note: The outputs C11 and C12 are 2x2 matrices.
    # C11 is at rows 0-1, cols 0-1 of C.
    # C12 is at rows 0-1, cols 2-3 of C.
    
    add_rank7_2x2(0, 0, 0, 0, 0, 0) # A11 * B11 -> C11 part 1
    add_rank7_2x2(0, 2, 2, 0, 0, 0) # A12 * B21 -> C11 part 2
    add_rank7_2x2(0, 0, 0, 2, 0, 2) # A11 * B12 -> C12 part 1
    add_rank7_2x2(0, 2, 2, 2, 0, 2) # A12 * B22 -> C12 part 2
    
    # Block 3: A_bot (row 2) * B_2 (cols 4-7)
    for j in range(4, 8):
        for k in range(4):
            U[2][col] = 1
            V[k * 8 + j][col] = 1
            W[(j) * 3 + 2][col] = 1
            col += 1
            
    # Block 4: A_top (rows 0-1) * B_2 (cols 4-7)
    # B_2 = [[B31, B32], [B41, B42]]
    # B31: rows 0-1, cols 4-5
    # B32: rows 0-1, cols 6-7
    # B41: rows 2-3, cols 4-5
    # B42: rows 2-3, cols 6-7
    
    # C_top2 = [[C21, C22]]
    # C21 = A11*B31 + A12*B41
    # C22 = A11*B32 + A12*B42
    
    add_rank7_2x2(0, 0, 0, 4, 0, 4) # A11 * B31 -> C21 part 1
    add_rank7_2x2(0, 2, 2, 4, 0, 4) # A12 * B41 -> C21 part 2
    add_rank7_2x2(0, 0, 0, 6, 0, 6) # A11 * B32 -> C22 part 1
    add_rank7_2x2(0, 2, 2, 6, 0, 6) # A12 * B42 -> C22 part 2

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
