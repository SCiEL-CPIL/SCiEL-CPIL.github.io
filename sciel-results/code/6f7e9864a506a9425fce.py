"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We decompose M_{3,4,8} into:
    # C1: M_{2,4,8} (rows 0,1 of A)
    # C2: M_{1,4,8} (row 2 of A)
    
    # C2: M_{1,4,8} has rank 1
    # U_C2 shape (4, 1), V_C2 shape (32, 1), W_C2 shape (8, 1)
    # All entries 1.
    
    # C1: M_{2,4,8}
    # Split B into B1 (4x4) and B2 (4x4)
    # C1 = [A1 B1, A1 B2]
    # A1 is 2x4.
    # M_{2,4,4} rank 28 (4 * Strassen 7)
    
    # Total rank = 1 + 28 + 28 = 57
    
    rank = 57
    U = [[0] * rank for _ in range(n * m)] # 12 x 57
    V = [[0] * rank for _ in range(m * p)] # 32 x 57
    W = [[0] * rank for _ in range(p * n)] # 24 x 57
    
    # Helper to add Strassen decomposition
    # Strassen for 2x2 x 2
    # U_strassen shape (4, 7)
    # V_strassen shape (4, 7)
    # W_strassen shape (4, 7)
    
    # Standard Strassen matrices (indices 0..3 for 2x2)
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # U rows: a00, a01, a10, a11
    # V rows: b00, b01, b10, b11
    # W rows: c00, c01, c10, c11
    
    U_str = [
        [1, 0, 1, 0, 1, -1, 0], # a00
        [0, 1, 0, 0, 1, 0, -1], # a01
        [0, 1, 0, 1, 0, 1, 0],  # a10
        [1, 0, -1, 1, 0, 0, 1]  # a11
    ]
    
    V_str = [
        [1, 1, 0, -1, 0, 1, 0], # b00
        [1, 0, 1, 0, 1, 1, 0],  # b01
        [1, 0, 0, 1, 0, 0, 1],  # b10
        [1, -1, -1, 0, 1, 0, 1] # b11
    ]
    
    W_str = [
        [1, -1, -1, 1, 0, 0, 0], # c00
        [0, 0, 1, 0, 1, 1, 0],   # c01
        [0, 0, 0, 1, 0, 1, 1],   # c10
        [1, -1, 0, -1, 1, 0, 0]  # c11
    ]
    
    # Function to add a 2x2x2 Strassen block
    def add_strassen(i_offset, j_offset, k_offset, col_start):
        # i_offset: row offset in A (0..1)
        # j_offset: col offset in A / row offset in B (0..3)
        # k_offset: col offset in B (0..7)
        
        for r in range(7):
            col = col_start + r
            
            # U: A indices
            for di in range(2):
                for dj in range(2):
                    i = i_offset + di
                    j = j_offset + dj
                    U[i * m + j][col] += U_str[di * 2 + dj][r]
            
            # V: B indices
            for dj in range(2):
                for dk in range(2):
                    j = j_offset + dj
                    k = k_offset + dk
                    V[j * p + k][col] += V_str[dj * 2 + dk][r]
            
            # W: C indices
            for dk in range(2):
                for di in range(2):
                    k = k_offset + dk
                    i = i_offset + di
                    W[k * n + i][col] += W_str[dk * 2 + di][r]
    
    # C1: A1 (rows 0,1) * B1 (cols 0,3)
    # A1 = [A11, A12] (2x2 each)
    # B1 = [[B11, B12], [B21, B22]] (2x2 each)
    # C1_left = A11 B11 + A12 B21
    # C1_right = A11 B12 + A12 B22
    
    # P1 = A11 B11
    # A11: i 0,1; j 0,1
    # B11: j 0,1; k 0,1
    # C: i 0,1; k 0,1
    add_strassen(0, 0, 0, 0)
    
    # P2 = A12 B21
    # A12: i 0,1; j 2,3
    # B21: j 2,3; k 0,1
    # C: i 0,1; k 0,1
    add_strassen(0, 2, 0, 7)
    
    # P3 = A11 B12
    # A11: i 0,1; j 0,1
    # B12: j 0,1; k 2,3
    # C: i 0,1; k 2,3
    add_strassen(0, 0, 2, 14)
    
    # P4 = A12 B22
    # A12: i 0,1; j 2,3
    # B22: j 2,3; k 2,3
    # C: i 0,1; k 2,3
    add_strassen(0, 2, 2, 21)
    
    # C1: A1 * B2 (cols 4,7)
    # B2 = [[B13, B14], [B23, B24]]
    # C1_left2 = A11 B13 + A12 B23
    # C1_right2 = A11 B14 + A12 B24
    
    # P5 = A11 B13
    # A11: i 0,1; j 0,1
    # B13: j 0,1; k 4,5
    # C: i 0,1; k 4,5
    add_strassen(0, 0, 4, 28)
    
    # P6 = A12 B23
    # A12: i 0,1; j 2,3
    # B23: j 2,3; k 4,5
    # C: i 0,1; k 4,5
    add_strassen(0, 2, 4, 35)
    
    # P7 = A11 B14
    # A11: i 0,1; j 0,1
    # B14: j 0,1; k 6,7
    # C: i 0,1; k 6,7
    add_strassen(0, 0, 6, 42)
    
    # P8 = A12 B24
    # A12: i 0,1; j 2,3
    # B24: j 2,3; k 6,7
    # C: i 0,1; k 6,7
    add_strassen(0, 2, 6, 49)
    
    # C2: A2 (row 2) * B (cols 0,7)
    # Rank 1
    # U: j 0..3
    # V: j 0..3, k 0..7
    # W: k 0..7
    
    col = 56
    for j in range(4):
        U[2 * m + j][col] = 1
    for j in range(4):
        for k in range(8):
            V[j * p + k][col] = 1
    for k in range(8):
        W[k * n + 2][col] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
