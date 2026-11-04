"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 4
    rank = 22
    U = [[0] * rank for _ in range(n * m)]  # 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 12 rows
    W = [[0] * rank for _ in range(p * n)]  # 8 rows

    # Strassen rank-7 decomposition of 2x2 * 2x2
    # Rows of U: (i,j) for i in {0,1}, j in {0,1}
    # Rows of V: (j,k) for j in {0,1}, k in {0,1}
    # Rows of W: (k,i) for k in {0,1}, i in {0,1}
    # Strassen formulas:
    # M1 = (A11)(B12 - B22)
    # M2 = (A11 + A21)(B22)
    # M3 = (A21 - A11)(B11)
    # M4 = (A22)(B21 - B11)
    # M5 = (A11 + A22)(B11 + B22)
    # M6 = (A12 - A22)(B11 + B21)
    # M7 = (A21 + A22)(B12 + B22)
    # C11 = M5 + M4 - M2 - M7
    # C12 = M1 + M7
    # C21 = M3 + M5
    # C22 = M1 + M2 - M4 - M3

    strassen = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # M1
        ([(0, 1)], [(0, 1), (1, 0)], [(0, 0), (1, 0)]),  # C11 += M1, C12 += M1
        # M2
        ([(0, 1), (1, 1)], [(1, 1)], [(0, 0), (1, 1)]),  # C11 -= M2, C22 += M2
        # M3
        ([(1, 1), (0, 1)], [(0, 0)], [(0, 1), (1, 0)]),  # C21 += M3, C22 -= M3
        # M4
        ([(1, 1)], [(1, 0), (0, 0)], [(1, 0), (0, 1)]),  # C11 -= M4, C22 -= M4
        # M5
        ([(0, 1), (1, 1)], [(0, 0), (1, 1)], [(0, 0), (0, 1), (1, 0), (1, 1)]),  # C11 += M5, C21 += M5
        # M6
        ([(0, 0), (1, 1)], [(0, 0), (1, 0)], []),  # M6 is for C12... wait let me recheck
        # Actually M6 doesn't appear in the standard C formulas above. Let me re-derive.
    ]

    # Let me be more careful with Strassen.
    # Standard Strassen:
    # M1 = (a11)(b12 - b22)
    # M2 = (a11 + a21)(b22)
    # M3 = (a21 - a11)(b11)
    # M4 = (a22)(b21 - b11)
    # M5 = (a11 + a22)(b11 + b22)
    # M6 = (a12 - a22)(b11 + b21)
    # M7 = (a21 + a22)(b12 + b22)
    # c11 = M5 + M4 - M2 - M7
    # c12 = M1 + M7
    # c21 = M3 + M5
    # c22 = M1 + M2 - M4 - M3

    # For M_{2,2,4}, split B into two 2x2 blocks: B_left (cols 0,1) and B_right (cols 2,3)
    # C_left = A * B_left (2x2), C_right = A * B_right (2x2)
    # So we do Strassen twice.

    # Block 1: cols 0,1 (ranks 0-6)
    # Block 2: cols 2,3 (ranks 7-13)
    # Block 3: row 2 of A (ranks 14-21) - trivial

    # Strassen helper for a given column offset
    def strassen_block(col_offset, rank_offset):
        """Add Strassen 7 ranks for multiplying 2x2 A by 2x2 B at columns col_offset, col_offset+1."""
        r = rank_offset
        # M1 = a11 * (b12 - b22)
        U[r][0] = 1  # a11 is at row (0,0) = 0
        V[r][col_offset + 1] = 1    # b12: j=0, k=1
        V[r][2 * (col_offset + 1)] = -1  # b22: j=1, k=1
        W[r][0] = 1  # c11 += M1
        W[r][1] = 1  # c12 += M1

        # M2 = (a11 + a21) * b22
        U[r + 1][0] = 1   # a11
        U[r + 1][3] = 1   # a21: row (1,1) = 3
        V[r + 1][2 * (col_offset + 1)] = 1  # b22
        W[r + 1][0] = -1  # c11 -= M2
        W[r + 1][3] = 1   # c22 += M2

        # M3 = (a21 - a11) * b11
        U[r + 2][3] = 1   # a21
        U[r + 2][0] = -1  # a11
        V[r + 2][col_offset] = 1  # b11: j=0, k=0
        W[r + 2][2] = 1   # c21 += M3
        W[r + 2][3] = -1  # c22 -= M3

        # M4 = a22 * (b21 - b11)
        U[r + 3][4] = 1   # a22: row (1,1) = 4... wait, row (i,j) = i*m+j = 1*2+1 = 3
        # Hmm, I need to be careful. Row index in U is i*m + j where i is row of A, j is col of A.
        # For 2x2 A: rows are 0,1 (i=0) and 2,3 (i=1). So a11=U[0], a12=U[1], a21=U[2], a22=U[3].
        # Wait, m=2 for the sub-block, but in the full problem m=3. Let me re-index.

        # Actually, for the full 2x3 * 3x4 problem:
        # U has 6 rows: (i,j) for i in {0,1}, j in {0,1,2}
        #   Row 0: (0,0), Row 1: (0,1), Row 2: (0,2)
        #   Row 3: (1,0), Row 4: (1,1), Row 5: (1,2)
        # V has 12 rows: (j,k) for j in {0,1,2}, k in {0,1,2,3}
        #   Rows 0-3: j=0, Rows 4-7: j=1, Rows 8-11: j=2
        # W has 8 rows: (k,i) for k in {0,1,2,3}, i in {0,1}
        #   Rows 0,1: k=0; Rows 2,3: k=1; Rows 4,5: k=2; Rows 6,7: k=3

        # For the 2x2 sub-block using columns 0,1 of B (and rows 0,1 of A):
        # a11 = U[0], a12 = U[1], a21 = U[3], a22 = U[4]
        # b11 = V[0], b12 = V[1], b21 = V[4], b22 = V[5]

        # Let me redo this properly.
    # I'll just hardcode all 22 ranks directly.

    # Reset
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] (ranks 0-6)
    # U rows: 0=(0,0), 1=(0,1), 3=(1,0), 4=(1,1)
    # V rows: 0=(0,0), 1=(0,1), 4=(1,0), 5=(1,1)
    # W rows: 0=(0,0), 1=(1,0), 2=(0,1), 3=(1,1)

    # M1 = a11*(b12-b22)
    U[0][0] = 1; V[1][0] = 1; V[5][0] = -1; W[0][0] = 1; W[2][0] = 1
    # M2 = (a11+a21)*b22
    U[0][1] = 1; U[3][1] = 1; V[5][1] = 1; W[0][1] = -1; W[3][1] = 1
    # M3 = (a21-a11)*b11
    U[3][2] = 1; U[0][2] = -1; V[0][2] = 1; W[1][2] = 1; W[3][2] = -1
    # M4 = a22*(b21-b11)
    U[4][3] = 1; V[4][3] = 1; V[0][3] = -1; W[0][3] = -1; W[3][3] = -1
    # M5 = (a11+a22)*(b11+b22)
    U[0][4] = 1; U[4][4] = 1; V[0][4] = 1; V[5][4] = 1; W[0][4] = 1; W[1][4] = 1; W[2][4] = 1; W[3][4] = 1
    # M6 = (a12-a22)*(b11+b21)
    U[1][5] = 1; U[4][5] = -1; V[0][5] = 1; V[4][5] = 1; W[2][5] = 1
    # M7 = (a21+a22)*(b12+b22)
    U[3][6] = 1; U[4][6] = 1; V[1][6] = 1; V[5][6] = 1; W[2][6] = 1; W[3][6] = -1

    # Strassen for A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4] (ranks 7-13)
    # V rows: 2=(0,2), 3=(0,3), 6=(1,2), 7=(1,3)
    # W rows: 4=(2,0), 5=(2,1), 6=(3,0), 7=(3,1)

    # M1
    U[0][7] = 1; V[3][7] = 1; V[7][7] = -1; W[4][7] = 1; W[6][7] = 1
    # M2
    U[0][8] = 1; U[3][8] = 1; V[7][8] = 1; W[4][8] = -1; W[7][8] = 1
    # M3
    U[3][9] = 1; U[0][9] = -1; V[2][9] = 1; W[5][9] = 1; W[7][9] = -1
    # M4
    U[4][10] = 1; V[6][10] = 1; V[2][10] = -1; W[4][10] = -1; W[7][10] = -1
    # M5
    U[0][11] = 1; U[4][11] = 1; V[2][11] = 1; V[7][11] = 1; W[4][11] = 1; W[5][11] = 1; W[6][11] = 1; W[7][11] = 1
    # M6
    U[1][12] = 1; U[4][12] = -1; V[2][12] = 1; V[6][12] = 1; W[6][12] = 1
    # M7
    U[3][13] = 1; U[4][13] = 1; V[3][13] = 1; V[7][13] = 1; W[6][13] = 1; W[7][13] = -1

    # Trivial: A[0:2,2] * B[2,0:4] -> C[0:2,0:4] (ranks 14-21)
    # U row: 2=(0,2), 5=(1,2)
    # V rows: 8=(2,0), 9=(2,1), 10=(2,2), 11=(2,3)
    # W rows: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1), 4=(2,0), 5=(2,1), 6=(3,0), 7=(3,1)

    # For each (i,k) pair, one rank-1 term
    idx = 0
    for i in range(2):
        for k in range(4):
            r = 14 + idx
            U[i * 3 + 2][r] = 1  # a_i2
            V[2 * 4 + k][r] = 1  # b_2k
            W[k * 2 + i][r] = 1  # c_ki
            idx += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
