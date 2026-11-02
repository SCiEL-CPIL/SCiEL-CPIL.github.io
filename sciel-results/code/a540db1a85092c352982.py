import time
import sys

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # Decomposition of 5x5 matrix multiplication into blocks of sizes 2, 2, 1.
    # We use Strassen (rank 7) for 2x2x2, and schoolbook (rank 4) for 2x1x1 or 1x2x2 etc.
    # Total rank calculated as 117.
    
    n, m, p = DIMENSIONS
    rank = 117
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val_u=1, val_v=1, val_w=1):
        nonlocal col
        U[u_idx][col] += val_u
        V[v_idx][col] += val_v
        W[w_idx][col] += val_w
        col += 1

    # Block definitions
    # A is 5x5, indices 0..4
    # B is 5x5, indices 0..4
    # C is 5x5, indices 0..4
    
    # Sizes:
    # A11: rows 0-1, cols 0-1 (2x2)
    # A12: rows 0-1, cols 2-3 (2x2)
    # A13: rows 0-1, cols 4 (2x1)
    # A21: rows 2-3, cols 0-1 (2x2)
    # A22: rows 2-3, cols 2-3 (2x2)
    # A23: rows 2-3, cols 4 (2x1)
    # A31: rows 4, cols 0-1 (1x2)
    # A32: rows 4, cols 2-3 (1x2)
    # A33: rows 4, cols 4 (1x1)
    
    # B11: rows 0-1, cols 0-1 (2x2)
    # B12: rows 0-1, cols 2-3 (2x2)
    # B13: rows 0-1, cols 4 (2x1)
    # B21: rows 2-3, cols 0-1 (2x2)
    # B22: rows 2-3, cols 2-3 (2x2)
    # B23: rows 2-3, cols 4 (2x1)
    # B31: rows 4, cols 0-1 (1x2)
    # B32: rows 4, cols 2-3 (1x2)
    # B33: rows 4, cols 4 (1x1)
    
    # C11: rows 0-1, cols 0-1 (2x2)
    # C12: rows 0-1, cols 2-3 (2x2)
    # C13: rows 0-1, cols 4 (2x1)
    # C21: rows 2-3, cols 0-1 (2x2)
    # C22: rows 2-3, cols 2-3 (2x2)
    # C23: rows 2-3, cols 4 (2x1)
    # C31: rows 4, cols 0-1 (1x2)
    # C32: rows 4, cols 2-3 (1x2)
    # C33: rows 4, cols 4 (1x1)
    
    # Strassen decomposition for 2x2x2
    # Inputs: A (2x2), B (2x2)
    # A indices: r1c1, r1c2, r2c1, r2c2
    # B indices: c1r1, c1r2, c2r1, c2r2
    # Output C: r1c1, r1c2, r2c1, r2c2
    # M1 = (A11+A22)(B11+B22) -> C11+=M1, C12-=M1, C21-=M1, C22+=M1
    # M2 = (A21+A22)B11 -> C21+=M2, C22-=M2
    # M3 = A11(B21-B22) -> C11-=M3, C12+=M3
    # M4 = A22(B12-B11) -> C21+=M4, C22+=M4
    # M5 = (A11+A12)B22 -> C11+=M5, C12-=M5
    # M6 = (A21-A11)(B11+B12) -> C21-=M6, C22+=M6
    # M7 = (A12-A22)(B21+B22) -> C12+=M7, C21+=M7
    
    def add_strassen(A_r1c1, A_r1c2, A_r2c1, A_r2c2, 
                     B_c1r1, B_c1r2, B_c2r1, B_c2r2,
                     C_r1c1, C_r1c2, C_r2c1, C_r2c2):
        nonlocal col
        # M1
        U[A_r1c1][col] = 1; U[A_r2c2][col] = 1
        V[B_c1r1][col] = 1; V[B_c2r2][col] = 1
        W[C_r1c1][col] = 1; W[C_r1c2][col] = -1; W[C_r2c1][col] = -1; W[C_r2c2][col] = 1
        col += 1
        # M2
        U[A_r2c1][col] = 1; U[A_r2c2][col] = 1
        V[B_c1r1][col] = 1
        W[C_r2c1][col] = 1; W[C_r2c2][col] = -1
        col += 1
        # M3
        U[A_r1c1][col] = 1
        V[B_c2r1][col] = 1; V[B_c2r2][col] = -1
        W[C_r1c1][col] = -1; W[C_r1c2][col] = 1
        col += 1
        # M4
        U[A_r2c2][col] = 1
        V[B_c1r2][col] = 1; V[B_c1r1][col] = -1
        W[C_r2c1][col] = 1; W[C_r2c2][col] = 1
        col += 1
        # M5
        U[A_r1c1][col] = 1; U[A_r1c2][col] = 1
        V[B_c2r2][col] = 1
        W[C_r1c1][col] = 1; W[C_r1c2][col] = -1
        col += 1
        # M6
        U[A_r2c1][col] = 1; U[A_r1c1][col] = -1
        V[B_c1r1][col] = 1; V[B_c1r2][col] = 1
        W[C_r2c1][col] = -1; W[C_r2c2][col] = 1
        col += 1
        # M7
        U[A_r1c2][col] = 1; U[A_r2c2][col] = -1
        V[B_c2r1][col] = 1; V[B_c2r2][col] = 1
        W[C_r1c2][col] = 1; W[C_r2c1][col] = 1
        col += 1

    def add_2x1x1(A_r1, A_r2, B_c, C_r1, C_r2):
        # A is 2x1, B is 1x1, C is 2x1
        # C1 = A1 B, C2 = A2 B
        nonlocal col
        U[A_r1][col] = 1
        V[B_c][col] = 1
        W[C_r1][col] = 1
        col += 1
        U[A_r2][col] = 1
        V[B_c][col] = 1
        W[C_r2][col] = 1
        col += 1

    def add_1x2x2(A_r, B_c1, B_c2, C_c1, C_c2):
        # A is 1x2, B is 2x2, C is 1x2
        # C1 = A1 B11 + A2 B21
        # C2 = A1 B12 + A2 B22
        # This is 4 scalar mults.
        nonlocal col
        U[A_r][col] = 1 # A1
        V[B_c1][col] = 1 # B11
        W[C_c1][col] = 1
        col += 1
        U[A_r][col] = 1 # A1
        V[B_c2][col] = 1 # B12
        W[C_c2][col] = 1
        col += 1
        U[A_r][col] = 1 # A2 (Wait, A is 1x2, so indices are A_r1, A_r2? No, A is row vector)
        # A indices: A_r, col 0 and A_r, col 1.
        # In our flat index, A_r is the row. The columns are separate indices in U?
        # U is (n*m, rank). So U[i*m+j, c].
        # For A (5x5), row 4, col 0 is index 4*5+0 = 20. Row 4, col 1 is 21.
        # So we need to pass specific indices.
        
    # Let's refine the helper functions to take flat indices.
    
    # Indices for A (5x5):
    # A11: 0,1,5,6
    # A12: 2,3,7,8
    # A13: 4,9
    # A21: 10,11,15,16
    # A22: 12,13,17,18
    # A23: 14,19
    # A31: 20,21
    # A32: 22,23
    # A33: 24
    
    # Indices for B (5x5):
    # B11: 0,1,5,6
    # B12: 2,3,7,8
    # B13: 4,9
    # B21: 10,11,15,16
    # B22: 12,13,17,18
    # B23: 14,19
    # B31: 20,21
    # B32: 22,23
    # B33: 24
    
    # Indices for C (5x5):
    # C11: 0,1,5,6
    # C12: 2,3,7,8
    # C13: 4,9
    # C21: 10,11,15,16
    # C22: 12,13,17,18
    # C23: 14,19
    # C31: 20,21
    # C32: 22,23
    # C33: 24
    
    # C11 (2x2) = A11 B11 + A12 B21 + A13 B31
    add_strassen(0, 1, 5, 6, 0, 5, 1, 6, 0, 1, 5, 6) # A11 B11
    add_strassen(2, 3, 7, 8, 10, 15, 11, 16, 0, 1, 5, 6) # A12 B21
    # A13 B31: A13 is (4,9), B31 is (20,21). C13 is (4,9)? No, C11 is (0,1,5,6).
    # A13 (2x1) * B31 (1x2) -> C11 (2x2).
    # Terms: A13[0]*B31[0] -> C11[0], A13[0]*B31[1] -> C11[1]
    #         A13[1]*B31[0] -> C11[5], A13[1]*B31[1] -> C11[6]
    U[4][col] = 1; V[20][col] = 1; W[0][col] = 1; col += 1
    U[4][col] = 1; V[21][col] = 1; W[1][col] = 1; col += 1
    U[9][col] = 1; V[20][col] = 1; W[5][col] = 1; col += 1
    U[9][col] = 1; V[21][col] = 1; W[6][col] = 1; col += 1
    
    # C12 (2x2) = A11 B12 + A12 B22 + A13 B32
    add_strassen(0, 1, 5, 6, 2, 7, 3, 8, 2, 3, 7, 8) # A11 B12
    add_strassen(2, 3, 7, 8, 12, 17, 13, 18, 2, 3, 7, 8) # A12 B22
    # A13 B32: A13 (4,9), B32 (22,23). C12 (2,3,7,8).
    U[4][col] = 1; V[22][col] = 1; W[2][col] = 1; col += 1
    U[4][col] = 1; V[23][col] = 1; W[3][col] = 1; col += 1
    U[9][col] = 1; V[22][col] = 1; W[7][col] = 1; col += 1
    U[9][col] = 1; V[23][col] = 1; W[8][col] = 1; col += 1
    
    # C13 (2x1) = A11 B13 + A12 B23 + A13 B33
    # A11 (2x2) B13 (2x1) -> C13 (2x1). Rank 4.
    U[0][col] = 1; V[4][col] = 1; W[4][col] = 1; col += 1
    U[0][col] = 1; V[9][col] = 1; W[9][col] = 1; col += 1
    U[5][col] = 1; V[4][col] = 1; W[4][col] = 1; col += 1
    U[5][col] = 1; V[9][col] = 1; W[9][col] = 1; col += 1
    # A12 (2x2) B23 (2x1) -> C13 (2x1). Rank 4.
    U[2][col] = 1; V[14][col] = 1; W[4][col] = 1; col += 1
    U[2][col] = 1; V[19][col] = 1; W[9][col] = 1; col += 1
    U[7][col] = 1; V[14][col] = 1; W[4][col] = 1; col += 1
    U[7][col] = 1; V[19][col] = 1; W[9][col] = 1; col += 1
    # A13 (2x1) B33 (1x1) -> C13 (2x1). Rank 2.
    U[4][col] = 1; V[24][col] = 1; W[4][col] = 1; col += 1
    U[9][col] = 1; V[24][col] = 1; W[9][col] = 1; col += 1
    
    # C21 (2x2) = A21 B11 + A22 B21 + A23 B31
    add_strassen(10, 11, 15, 16, 0, 5, 1, 6, 10, 11, 15, 16) # A21 B11
    add_strassen(12, 13, 17, 18, 10, 15, 11, 16, 10, 11, 15, 16) # A22 B21
    # A23 B31: A23 (14,19), B31 (20,21). C21 (10,11,15,16).
    U[14][col] = 1; V[20][col] = 1; W[10][col] = 1; col += 1
    U[14][col] = 1; V[21][col] = 1; W[11][col] = 1; col += 1
    U[19][col] = 1; V[20][col] = 1; W[15][col] = 1; col += 1
    U[19][col] = 1; V[21][col] = 1; W[16][col] = 1; col += 1
    
    # C22 (2x2) = A21 B12 + A22 B22 + A23 B32
    add_strassen(10, 11, 15, 16, 2, 7, 3, 8, 12, 13, 17, 18) # A21 B12
    add_strassen(12, 13, 17, 18, 12, 17, 13, 18, 12, 13, 17, 18) # A22 B22
    # A23 B32: A23 (14,19), B32 (22,23). C22 (12,13,17,18).
    U[14][col] = 1; V[22][col] = 1; W[12][col] = 1; col += 1
    U[14][col] = 1; V[23][col] = 1; W[13][col] = 1; col += 1
    U[19][col] = 1; V[22][col] = 1; W[17][col] = 1; col += 1
    U[19][col] = 1; V[23][col] = 1; W[18][col] = 1; col += 1
    
    # C23 (2x1) = A21 B13 + A22 B23 + A23 B33
    # A21 (2x2) B13 (2x1) -> C23 (2x1). Rank 4.
    U[10][col] = 1; V[4][col] = 1; W[14][col] = 1; col += 1
    U[10][col] = 1; V[9][col] = 1; W[19][col] = 1; col += 1
    U[15][col] = 1; V[4][col] = 1; W[14][col] = 1; col += 1
    U[15][col] = 1; V[9][col] = 1; W[19][col] = 1; col += 1
    # A22 (2x2) B23 (2x1) -> C23 (2x1). Rank 4.
    U[12][col] = 1; V[14][col] = 1; W[14][col] = 1; col += 1
    U[12][col] = 1; V[19][col] = 1; W[19][col] = 1; col += 1
    U[17][col] = 1; V[14][col] = 1; W[14][col] = 1; col += 1
    U[17][col] = 1; V[19][col] = 1; W[19][col] = 1; col += 1
    # A23 (2x1) B33 (1x1) -> C23 (2x1). Rank 2.
    U[14][col] = 1; V[24][col] = 1; W[14][col] = 1; col += 1
    U[19][col] = 1; V[24][col] = 1; W[19][col] = 1; col += 1
    
    # C31 (1x2) = A31 B11 + A32 B21 + A33 B31
    # A31 (1x2) B11 (2x2) -> C31 (1x2). Rank 4.
    U[20][col] = 1; V[0][col] = 1; W[20][col] = 1; col += 1
    U[20][col] = 1; V[5][col] = 1; W[21][col] = 1; col += 1
    U[21][col] = 1; V[0][col] = 1; W[20][col] = 1; col += 1
    U[21][col] = 1; V[5][col] = 1; W[21][col] = 1; col += 1
    # A32 (1x2) B21 (2x2) -> C31 (1x2). Rank 4.
    U[22][col] = 1; V[10][col] = 1; W[20][col] = 1; col += 1
    U[22][col] = 1; V[15][col] = 1; W[21][col] = 1; col += 1
    U[23][col] = 1; V[10][col] = 1; W[20][col] = 1; col += 1
    U[23][col] = 1; V[15][col] = 1; W[21][col] = 1; col += 1
    # A33 (1x1) B31 (1x2) -> C31 (1x2). Rank 2.
    U[24][col] = 1; V[20][col] = 1; W[20][col] = 1; col += 1
    U[24][col] = 1; V[21][col] = 1; W[21][col] = 1; col += 1
    
    # C32 (1x2) = A31 B12 + A32 B22 + A33 B32
    # A31 (1x2) B12 (2x2) -> C32 (1x2). Rank 4.
    U[20][col] = 1; V[2][col] = 1; W[22][col] = 1; col += 1
    U[20][col] = 1; V[7][col] = 1; W[23][col] = 1; col += 1
    U[21][col] = 1; V[2][col] = 1; W[22][col] = 1; col += 1
    U[21][col] = 1; V[7][col] = 1; W[23][col] = 1; col += 1
    # A32 (1x2) B22 (2x2) -> C32 (1x2). Rank 4.
    U[22][col] = 1; V[12][col] = 1; W[22][col] = 1; col += 1
    U[22][col] = 1; V[17][col] = 1; W[23][col] = 1; col += 1
    U[23][col] = 1; V[12][col] = 1; W[22][col] = 1; col += 1
    U[23][col] = 1; V[17][col] = 1; W[23][col] = 1; col += 1
    # A33 (1x1) B32 (1x2) -> C32 (1x2). Rank 2.
    U[24][col] = 1; V[22][col] = 1; W[22][col] = 1; col += 1
    U[24][col] = 1; V[23][col] = 1; W[23][col] = 1; col += 1
    
    # C33 (1x1) = A31 B13 + A32 B23 + A33 B33
    # A31 (1x2) B13 (2x1) -> C33 (1x1). Rank 2.
    U[20][col] = 1; V[4][col] = 1; W[24][col] = 1; col += 1
    U[21][col] = 1; V[9][col] = 1; W[24][col] = 1; col += 1
    # A32 (1x2) B23 (2x1) -> C33 (1x1). Rank 2.
    U[22][col] = 1; V[14][col] = 1; W[24][col] = 1; col += 1
    U[23][col] = 1; V[19][col] = 1; W[24][col] = 1; col += 1
    # A33 (1x1) B33 (1x1) -> C33 (1x1). Rank 1.
    U[24][col] = 1; V[24][col] = 1; W[24][col] = 1; col += 1
    
    # Verify col == rank
    if col != rank:
        # If mismatch, we might have miscounted.
        # Adjust rank to col to be safe, though it should be 117.
        rank = col
        # Truncate or pad
        if rank > 117:
             U = [row[:rank] for row in U]
             V = [row[:rank] for row in V]
             W = [row[:rank] for row in W]
        else:
             U = [row + [0]*(rank-len(row)) for row in U]
             V = [row + [0]*(rank-len(row)) for row in V]
             W = [row + [0]*(rank-len(row)) for row in W]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
