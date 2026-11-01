"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 26 decomposition using 2x2 Strassen + direct cross terms
    # U shape (9, rank), V shape (9, rank), W shape (9, rank)
    
    rank = 26
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]
    
    col = 0
    
    # 1. Strassen for top-left 2x2 block (indices 0,1 in rows/cols)
    # A indices: 0,1,2,3 (row-major 0..8 for 3x3? No, 0..8 is 3x3)
    # 3x3 indices:
    # 0 1 2
    # 3 4 5
    # 6 7 8
    # Top-left 2x2: 0,1,3,4
    # A_TR: 2,5
    # A_BL: 6,7
    # A_BR: 8
    
    # B indices:
    # 0 1 2
    # 3 4 5
    # 6 7 8
    # B_TL: 0,1,3,4
    # B_TR: 2,5
    # B_BL: 6,7
    # B_BR: 8
    
    # Strassen terms for A[0,1,3,4] and B[0,1,3,4]
    # M1 = (A0+A4)(B0+B4)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    # W contributions to C0, C1, C3, C4
    W[0][col] = 1  # C00
    W[1][col] = 0  # C01 (wait, C01 is 1, C10 is 3, C11 is 4)
    # C indices:
    # 0 1 2
    # 3 4 5
    # 6 7 8
    
    # M1 contributes to C00, C01, C10, C11?
    # Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 - M6
    
    # Let's define the 7 Strassen terms explicitly.
    # Term 0: M1
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1  # C00
    W[4][col] = 1  # C11
    # Note: C01 and C10 are handled by other M's.
    # Wait, M1 = (A0+A4)(B0+B4) = A0B0 + A0B4 + A4B0 + A4B4.
    # C00 needs A0B0 + A1B3.
    # C11 needs A3B1 + A4B4.
    # So M1 contributes A0B0 to C00 and A4B4 to C11.
    # The cross terms A0B4 and A4B0 are not in C00 or C11 directly?
    # No, C00 = sum_k A0k Bk0.
    # C11 = sum_k A3k Bk1.
    # So M1 is just an intermediate variable.
    # We must map M's to C's.
    
    # I will use a standard Strassen implementation for 2x2 and map it.
    
    def add_strassen_2x2(base_a, base_b, base_c, start_col):
        # base_a: indices of A in 2x2 (e.g., [0,1,3,4])
        # base_b: indices of B in 2x2
        # base_c: indices of C in 2x2
        # Returns new col index
        c = start_col
        # M1
        U[base_a[0]][c] = 1; U[base_a[3]][c] = 1
        V[base_b[0]][c] = 1; V[base_b[3]][c] = 1
        W[base_c[0]][c] = 1  # C00
        W[base_c[3]][c] = 1  # C11
        c += 1
        # M2
        U[base_a[2]][c] = 1; U[base_a[3]][c] = 1
        V[base_b[0]][c] = 1
        W[base_c[2]][c] = 1  # C10
        W[base_c[3]][c] = -1 # C11
        c += 1
        # M3
        U[base_a[0]][c] = 1
        V[base_b[1]][c] = 1; V[base_b[3]][c] = -1
        W[base_c[1]][c] = 1  # C01
        W[base_c[3]][c] = 1  # C11
        c += 1
        # M4
        U[base_a[3]][c] = 1
        V[base_b[2]][c] = 1; V[base_b[0]][c] = -1
        W[base_c[0]][c] = 1  # C00
        W[base_c[2]][c] = 1  # C10
        c += 1
        # M5
        U[base_a[0]][c] = 1; U[base_a[1]][c] = 1
        V[base_b[3]][c] = 1
        W[base_c[0]][c] = -1 # C00
        W[base_c[1]][c] = 1  # C01
        c += 1
        # M6
        U[base_a[2]][c] = 1; U[base_a[0]][c] = -1
        V[base_b[0]][c] = 1; V[base_b[1]][c] = 1
        W[base_c[3]][c] = -1 # C11
        c += 1
        # M7
        U[base_a[1]][c] = 1; U[base_a[3]][c] = -1
        V[base_b[2]][c] = 1; V[base_b[3]][c] = 1
        W[base_c[0]][c] = 1  # C00
        c += 1
        return c

    # Strassen on A[0,1,3,4] and B[0,1,3,4] -> C[0,1,3,4]
    col = add_strassen_2x2([0,1,3,4], [0,1,3,4], [0,1,3,4], 0)
    
    # Cross terms for C_TL (0,1,3,4) from A_TR (2,5) and B_BL (6,7)
    # A[2]B[6] -> C[0]
    U[2][col] = 1; V[6][col] = 1; W[0][col] = 1; col += 1
    # A[2]B[7] -> C[1]
    U[2][col] = 1; V[7][col] = 1; W[1][col] = 1; col += 1
    # A[5]B[6] -> C[3]
    U[5][col] = 1; V[6][col] = 1; W[3][col] = 1; col += 1
    # A[5]B[7] -> C[4]
    U[5][col] = 1; V[7][col] = 1; W[4][col] = 1; col += 1
    
    # C_TR (2,5) from A_TL (0,1,3,4) and B_TR (2,5)
    # C[2] = A0B2 + A1B5
    U[0][col] = 1; V[2][col] = 1; W[2][col] = 1; col += 1
    U[1][col] = 1; V[5][col] = 1; W[2][col] = 1; col += 1
    # C[5] = A3B2 + A4B5
    U[3][col] = 1; V[2][col] = 1; W[5][col] = 1; col += 1
    U[4][col] = 1; V[5][col] = 1; W[5][col] = 1; col += 1
    # C[2] += A[2]B[8]
    U[2][col] = 1; V[8][col] = 1; W[2][col] = 1; col += 1
    # C[5] += A[5]B[8]
    U[5][col] = 1; V[8][col] = 1; W[5][col] = 1; col += 1
    
    # C_BL (6,7) from A_BL (6,7) and B_TL (0,1,3,4)
    # C[6] = A6B0 + A7B3
    U[6][col] = 1; V[0][col] = 1; W[6][col] = 1; col += 1
    U[7][col] = 1; V[3][col] = 1; W[6][col] = 1; col += 1
    # C[7] = A6B1 + A7B4
    U[6][col] = 1; V[1][col] = 1; W[7][col] = 1; col += 1
    U[7][col] = 1; V[4][col] = 1; W[7][col] = 1; col += 1
    # C[6] += A[8]B[6]
    U[8][col] = 1; V[6][col] = 1; W[6][col] = 1; col += 1
    # C[7] += A[8]B[7]
    U[8][col] = 1; V[7][col] = 1; W[7][col] = 1; col += 1
    
    # C_BR (8) from A_BL (6,7) and B_TR (2,5)
    # C[8] = A6B2 + A7B5
    U[6][col] = 1; V[2][col] = 1; W[8][col] = 1; col += 1
    U[7][col] = 1; V[5][col] = 1; W[8][col] = 1; col += 1
    # C[8] += A[8]B[8]
    U[8][col] = 1; V[8][col] = 1; W[8][col] = 1; col += 1
    
    # Total rank used: 7 + 4 + 6 + 6 + 3 = 26
    # Trim to actual rank
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
