"""Matrix multiplication 3x3 * 3x8 via Strassen block decomposition, rank 68."""

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = 3, 3, 8
    rank = 68
    U = [[0] * rank for _ in range(n * m)]   # 9 x rank
    V = [[0] * rank for _ in range(m * p)]   # 24 x rank
    W = [[0] * rank for _ in range(p * n)]   # 24 x rank

    col = 0

    # Block structure:
    # A (3x3):
    #   A11: rows 0-1, cols 0-1 (2x2)
    #   A12: rows 0-1, col 2   (2x1)
    #   A21: row 2, cols 0-1   (1x2)
    #   A22: row 2, col 2      (1x1)
    #
    # B (3x8):
    #   B11: rows 0-1, cols 0-7 (2x8) -> split into 4 blocks of 2x2
    #   B21: row 2, cols 0-7    (1x8)
    #
    # C (3x8):
    #   C_top: rows 0-1, cols 0-7 (2x8) = A11*B11 + A12*B21
    #   C_bot: row 2, cols 0-7    (1x8) = A21*B11 + A22*B21

    # --- Part 1: A11 (2x2) * B11 (2x8), split B11 into four 2x2 blocks ---
    # For each of 4 column blocks (cols 0-1, 2-3, 4-5, 6-7 of B):
    for block in range(4):
        b_col_start = block * 2
        # Strassen for A11 (2x2) * B11_block (2x2)
        # A11 flat indices in U: (i*3+j) for i in 0,1; j in 0,1 -> 0,1,3,4
        # B11_block flat indices in V: (j*8+k) for j in 0,1; k in b_col_start..b_col_start+1
        #   j=0: k=b_col_start -> 0*8+b_col_start = b_col_start
        #   j=0: k=b_col_start+1 -> b_col_start+1
        #   j=1: k=b_col_start -> 8+b_col_start
        #   j=1: k=b_col_start+1 -> 8+b_col_start+1
        # C output: rows 0,1; cols b_col_start, b_col_start+1
        # C[i, j] stored in W[j*n + i] = W[j*3 + i]
        #   i=0, j=b_col_start -> W[b_col_start*3 + 0]
        #   i=0, j=b_col_start+1 -> W[(b_col_start+1)*3 + 0]
        #   i=1, j=b_col_start -> W[b_col_start*3 + 1]
        #   i=1, j=b_col_start+1 -> W[(b_col_start+1)*3 + 1]

        a11 = [0, 1, 3, 4]       # A11 flat: (0,0),(0,1),(1,0),(1,1)
        b11 = [b_col_start, b_col_start + 1, 8 + b_col_start, 8 + b_col_start + 1]  # B flat
        c_out = [b_col_start * 3 + 0, (b_col_start + 1) * 3 + 0,
                 b_col_start * 3 + 1, (b_col_start + 1) * 3 + 1]  # W flat for C rows 0,1

        # Strassen 7 tensors
        # M1 = (A00+A11)*(B00+B11)
        U[a11[0]][col] = 1
        U[a11[3]][col] = 1
        V[b11[0]][col] = 1
        V[b11[3]][col] = 1
        W[c_out[0]][col] = 1   # C11
        W[c_out[3]][col] = 1   # C22
        col += 1

        # M2 = (A10+A11)*B00
        U[a11[2]][col] = 1
        U[a11[3]][col] = 1
        V[b11[0]][col] = 1
        W[c_out[2]][col] = 1   # C21
        col += 1

        # M3 = A00*(B01-B11)
        U[a11[0]][col] = 1
        V[b11[1]][col] = 1
        V[b11[3]][col] = -1
        W[c_out[1]][col] = 1   # C12
        col += 1

        # M4 = A11*(B10-B00)
        U[a11[3]][col] = 1
        V[b11[2]][col] = 1
        V[b11[0]][col] = -1
        W[c_out[0]][col] = 1   # C11 (add)
        W[c_out[3]][col] = 1   # C22 (add)
        col += 1

        # M5 = (A00+A01)*B11
        U[a11[0]][col] = 1
        U[a11[1]][col] = 1
        V[b11[3]][col] = 1
        W[c_out[0]][col] = -1  # C11 (subtract)
        W[c_out[1]][col] = 1   # C12 (add)
        col += 1

        # M6 = (A10-A00)*(B00+B01)
        U[a11[2]][col] = 1
        U[a11[0]][col] = -1
        V[b11[0]][col] = 1
        V[b11[1]][col] = 1
        W[c_out[3]][col] = 1   # C22 (add)
        col += 1

        # M7 = (A01-A11)*(B10+B11)
        U[a11[1]][col] = 1
        U[a11[3]][col] = -1
        V[b11[2]][col] = 1
        V[b11[3]][col] = 1
        W[c_out[0]][col] = 1   # C11 (add)
        col += 1

    # --- Part 2: A12 (2x1) * B21 (1x8), rank 16 ---
    # A12: rows 0,1; col 2. Flat U: 0*3+2=2, 1*3+2=5
    # B21: row 2; cols 0..7. Flat V: 2*8+k = 16+k for k=0..7
    # C output: rows 0,1; cols 0..7. W[j*3+i] for i in 0,1; j in 0..7
    for k in range(8):
        for i in range(2):
            U[i * 3 + 2][col] = 1        # A12[i, 2]
            V[16 + k][col] = 1           # B21[2, k]
            W[k * 3 + i][col] = 1        # C[i, k]
            col += 1

    # --- Part 3: A21 (1x2) * B11 (2x8), rank 16 ---
    # A21: row 2; cols 0,1. Flat U: 2*3+0=6, 2*3+1=7
    # B11: rows 0,1; cols 0..7. Flat V: j*8+k for j in 0,1; k in 0..7
    # C output: row 2; cols 0..7. W[j*3+2] for j in 0..7
    for k in range(8):
        for j in range(2):
            U[2 * 3 + j][col] = 1        # A21[2, j]
            V[j * 8 + k][col] = 1        # B11[j, k]
            W[k * 3 + 2][col] = 1        # C[2, k]
            col += 1

    # --- Part 4: A22 (1x1) * B21 (1x8), rank 8 ---
    # A22: row 2, col 2. Flat U: 2*3+2=8
    # B21: row 2; cols 0..7. Flat V: 16+k
    # C output: row 2; cols 0..7. W[j*3+2] for j in 0..7
    for k in range(8):
        U[8][col] = 1
        V[16 + k][col] = 1
        W[k * 3 + 2][col] = 1
        col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"
    return {"U": U, "V": V, "W": W}
