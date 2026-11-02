"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=8
    # Strategy:
    # 1. Pad A (3x4) to A_pad (4x4) with a zero row at the bottom.
    # 2. Compute C_pad (4x8) = A_pad * B.
    # 3. Split A_pad into A_top (2x4) and A_bot (1x4, which is zero).
    #    Wait, A_bot is the 3rd row of the original A? No, we padded.
    #    Original A is 3x4.
    #    A_top = A[0:2, :] (2x4)
    #    A_mid = A[2:3, :] (1x4)
    #    A_pad = [A_top; A_mid; 0]
    #    We compute C_top (2x8) = A_top * B
    #    We compute C_mid (1x8) = A_mid * B
    #    C_bot (1x8) = 0 * B = 0
    #    We only return C_top and C_mid.
    
    # Rank of A_top * B (2x4 * 4x8):
    # Split B into B_L (4x4) and B_R (4x4).
    # A_top * B_L (2x4) and A_top * B_R (2x4).
    # Each 2x4 product uses 4 Strassen 2x2x2 steps (rank 7 each) -> rank 28.
    # Total for C_top = 28 + 28 = 56.
    
    # Rank of A_mid * B (1x4 * 4x8):
    # This is a 1x4x8 tensor. Rank 4.
    
    # Total rank = 56 + 4 = 60.
    
    rank = 60
    U = [[0] * rank for _ in range(n * m)] # 12 x 60
    V = [[0] * rank for _ in range(m * p)] # 32 x 60
    W = [[0] * rank for _ in range(p * n)] # 24 x 60
    
    # Helper to add Strassen 2x2x2 decomposition
    # Indices for 2x2x2:
    # A indices: (i,j) -> 0..3
    # B indices: (j,k) -> 0..3
    # C indices: (i,k) -> 0..3
    # Strassen M1..M7
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    def add_strassen_2x2x2(offset_u, offset_v, offset_w, col_start):
        # offset_u: index in U for A[0,0]
        # offset_v: index in V for B[0,0]
        # offset_w: index in W for C[0,0]
        
        # M1
        U[offset_u][col_start] = 1
        U[offset_u + 3][col_start] = 1
        V[offset_v][col_start] = 1
        V[offset_v + 3][col_start] = 1
        W[offset_w][col_start] = 1
        W[offset_w + 1][col_start] = 1
        W[offset_w + 2][col_start] = 1
        W[offset_w + 3][col_start] = 1
        
        # M2
        U[offset_u + 2][col_start + 1] = 1
        U[offset_u + 3][col_start + 1] = 1
        V[offset_v][col_start + 1] = 1
        W[offset_w + 2][col_start + 1] = 1
        W[offset_w + 3][col_start + 1] = 1
        
        # M3
        U[offset_u][col_start + 2] = 1
        V[offset_v + 2][col_start + 2] = 1
        V[offset_v + 3][col_start + 2] = -1
        W[offset_w + 1][col_start + 2] = 1
        W[offset_w + 2][col_start + 2] = 1
        
        # M4
        U[offset_u + 3][col_start + 3] = 1
        V[offset_v + 1][col_start + 3] = 1
        V[offset_v][col_start + 3] = -1
        W[offset_w][col_start + 3] = 1
        W[offset_w + 2][col_start + 3] = 1
        
        # M5
        U[offset_u][col_start + 4] = 1
        U[offset_u + 1][col_start + 4] = 1
        V[offset_v + 3][col_start + 4] = 1
        W[offset_w + 1][col_start + 4] = 1
        W[offset_w + 3][col_start + 4] = 1
        
        # M6
        U[offset_u + 2][col_start + 5] = 1
        U[offset_u][col_start + 5] = -1
        V[offset_v][col_start + 5] = 1
        V[offset_v + 1][col_start + 5] = 1
        W[offset_w + 2][col_start + 5] = 1
        W[offset_w + 3][col_start + 5] = 1
        
        # M7
        U[offset_u + 1][col_start + 6] = 1
        U[offset_u + 3][col_start + 6] = -1
        V[offset_v + 2][col_start + 6] = 1
        V[offset_v + 3][col_start + 6] = 1
        W[offset_w][col_start + 6] = 1
        W[offset_w + 1][col_start + 6] = -1
        
        return col_start + 7

    # A_top is 2x4.
    # A_top[0,0] is U[0], A_top[0,1] is U[1], A_top[0,2] is U[2], A_top[0,3] is U[3]
    # A_top[1,0] is U[4], A_top[1,1] is U[5], A_top[1,2] is U[6], A_top[1,3] is U[7]
    
    # B is 4x8.
    # B[0,0] is V[0], ..., B[0,7] is V[7]
    # B[1,0] is V[8], ...
    
    # C_top is 2x8.
    # C_top[0,0] is W[0], ..., C_top[0,7] is W[7]
    # C_top[1,0] is W[8], ...
    
    # We need to compute C_top = A_top * B.
    # Split B into B_L (cols 0-3) and B_R (cols 4-7).
    # Compute C_top_L = A_top * B_L (2x4)
    # Compute C_top_R = A_top * B_R (2x4)
    
    # A_top * B_L:
    # A_top is 2x4. B_L is 4x4.
    # Split A_top into A1 (2x2) and A2 (2x2).
    # A1: U[0..3], A2: U[4..7]
    # B_L: Split into 4 2x2 blocks.
    # B11: V[0..3], B12: V[4..7], B21: V[8..11], B22: V[12..15]
    # Result C_L is 2x4. Split into 4 2x2 blocks.
    # C11: W[0..3], C12: W[4..7], C21: W[8..11], C22: W[12..15]
    
    # We need to sum 4 Strassen products.
    # P1 = A1 * B11 -> C11
    # P2 = A2 * B21 -> C11
    # P3 = A1 * B12 -> C12
    # P4 = A2 * B22 -> C12
    # P5 = A1 * B21 -> C21
    # P6 = A2 * B22 -> C21
    # P7 = A1 * B12 -> C22
    # P8 = A2 * B22 -> C22
    
    # Wait, C11 = A1 B11 + A2 B21.
    # This requires 2 Strassen steps.
    # Step 1: A1 * B11.
    # Step 2: A2 * B21.
    # The results are added in W.
    
    col = 0
    # C11 part
    col = add_strassen_2x2x2(0, 0, 0, col) # A1 * B11 -> C11
    col = add_strassen_2x2x2(4, 8, 0, col) # A2 * B21 -> C11
    
    # C12 part
    col = add_strassen_2x2x2(0, 4, 4, col) # A1 * B12 -> C12
    col = add_strassen_2x2x2(4, 12, 4, col) # A2 * B22 -> C12
    
    # C21 part
    col = add_strassen_2x2x2(0, 8, 8, col) # A1 * B21 -> C21
    col = add_strassen_2x2x2(4, 12, 8, col) # A2 * B22 -> C21
    
    # C22 part
    col = add_strassen_2x2x2(0, 4, 12, col) # A1 * B12 -> C22
    col = add_strassen_2x2x2(4, 12, 12, col) # A2 * B22 -> C22
    
    # Now B_R (cols 4-7).
    # B_R11: V[16..19], B_R12: V[20..23], B_R21: V[24..27], B_R22: V[28..31]
    # Result C_R is 2x4.
    # C_R11: W[16..19], C_R12: W[20..23], C_R21: W[24..27], C_R22: W[28..31]
    
    col = add_strassen_2x2x2(0, 16, 16, col) # A1 * B_R11 -> C_R11
    col = add_strassen_2x2x2(4, 24, 16, col) # A2 * B_R21 -> C_R11
    
    col = add_strassen_2x2x2(0, 20, 20, col) # A1 * B_R12 -> C_R12
    col = add_strassen_2x2x2(4, 28, 20, col) # A2 * B_R22 -> C_R12
    
    col = add_strassen_2x2x2(0, 24, 24, col) # A1 * B_R21 -> C_R21
    col = add_strassen_2x2x2(4, 28, 24, col) # A2 * B_R22 -> C_R21
    
    col = add_strassen_2x2x2(0, 20, 28, col) # A1 * B_R12 -> C_R22
    col = add_strassen_2x2x2(4, 28, 28, col) # A2 * B_R22 -> C_R22
    
    # Now A_mid (1x4) * B (4x8).
    # A_mid is row 2 of A. U[8..11].
    # B is V[0..31].
    # C_mid is row 2 of C. W[16..23]?
    # C indices:
    # C_top is rows 0,1. W[0..15].
    # C_mid is row 2. W[16..23].
    # C_bot is row 3. W[24..31] (unused).
    
    # Rank 4 for 1x4x8.
    # U_mid: U[8], U[9], U[10], U[11].
    # V: V[0..31].
    # W_mid: W[16..23].
    
    # Term 1: A_mid[0] * B[0, :] -> C_mid[0..7]
    # U[8] * V[0..7] -> W[16..23]
    U[8][col] = 1
    for k in range(8):
        V[k][col] = 1
        W[16+k][col] = 1
    col += 1
    
    # Term 2: A_mid[1] * B[1, :] -> C_mid[0..7]
    U[9][col] = 1
    for k in range(8):
        V[8+k][col] = 1
        W[16+k][col] = 1
    col += 1
    
    # Term 3: A_mid[2] * B[2, :] -> C_mid[0..7]
    U[10][col] = 1
    for k in range(8):
        V[16+k][col] = 1
        W[16+k][col] = 1
    col += 1
    
    # Term 4: A_mid[3] * B[3, :] -> C_mid[0..7]
    U[11][col] = 1
    for k in range(8):
        V[24+k][col] = 1
        W[16+k][col] = 1
    col += 1
    
    # Verify col == rank
    assert col == rank, f"Rank mismatch: {col} != {rank}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
