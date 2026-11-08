"""Matrix multiplication 2x4 by 4x6: rank-minimal exact tensor decomposition."""

def construct():
    n, m, p = 2, 4, 6  # A is n x m, B is m x p
    U_rows = n * m  # 8
    V_rows = m * p  # 24
    W_rows = p * n  # 12

    # Strassen 2x2 decomposition:
    # Given A (2x2), B (2x2), compute C = A*B using 7 scalar multiplications:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # For each 2x2 block of B (columns 2k, 2k+1), we use one Strassen.
    # There are 3 such blocks (k=0,1,2), each covering 2 columns.
    # For each (i, j) pair in A where i in {0,1}, j in {0,1}, we have a 2x2 block.
    # But A is 2x4, so we have 2 blocks of 2x2 in A as well (columns 0-1 and 2-3).
    #
    # Actually, let me think more carefully.
    # A is 2x4, B is 4x6.
    # We can split B into 3 blocks of 4x2 (columns 0-1, 2-3, 4-5).
    # For each 4x2 block B_k, we can split it into two 2x2 blocks: B_k0 (rows 0-1) and B_k1 (rows 2-3).
    # A can be split into A0 (columns 0-1, 2x2) and A1 (columns 2-3, 2x2).
    #
    # C = A * B = A0 * B_top + A1 * B_bottom
    # where B_top is the top 2 rows of B (2x6), B_bottom is bottom 2 rows (2x6).
    #
    # For B_top (2x6), split into 3 blocks of 2x2.
    # For each 2x2 block, use Strassen: 7 terms.
    # So A0 * B_top uses 3 * 7 = 21 terms.
    # Similarly, A1 * B_bottom uses 3 * 7 = 21 terms.
    # Total: 42 terms.

    # Strassen coefficients:
    # For a 2x2 matrix A_sub = [[a00, a01], [a10, a11]] and B_sub = [[b00, b01], [b10, b11]]:
    # Each of the 7 M terms is a product of a linear combination of A entries and a linear combination of B entries.
    # In the tensor decomposition:
    #   U[i*m+j, c] contributes to A[i,j]
    #   V[j'*p+k, c] contributes to B[j',k]
    #   W[k'*n+i', c] contributes to C[i',k']

    # For A0 (columns 0,1): A_sub[i, j] = A[i, j] for i in {0,1}, j in {0,1}
    # For A1 (columns 2,3): A_sub[i, j] = A[i, j+2] for i in {0,1}, j in {0,1}

    # Strassen M terms for a single 2x2 block:
    # M1: (a00+a11) * (b00+b11)
    # M2: (a10+a11) * b00
    # M3: a00 * (b01-b11)
    # M4: a11 * (b10-b00)
    # M5: (a00+a01) * b11
    # M6: (a10-a00) * (b00+b01)
    # M7: (a01-a11) * (b10+b11)

    # Output:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Let's build the decomposition.
    # Each term c has a coefficient for each A entry, each B entry, and each C entry.

    # We'll store U, V, W as lists of lists.
    rank = 42
    U = [[0.0] * rank for _ in range(U_rows)]
    V = [[0.0] * rank for _ in range(V_rows)]
    W = [[0.0] * rank for _ in range(W_rows)]

    col = 0

    # Process A0 (A columns 0-1) times each 2x2 block of B_top (B rows 0-1, columns 2t-2t+1)
    # And A1 (A columns 2-3) times each 2x2 block of B_bottom (B rows 2-3, columns 2t-2t+1)

    for a_block in range(2):  # 0: A0 (cols 0-1), 1: A1 (cols 2-3)
        for t in range(3):  # 3 blocks of B, each 2x2
            # B block is rows 2*a_block to 2*a_block+1, cols 2*t to 2*t+1
            b_row0 = 2 * a_block
            b_row1 = 2 * a_block + 1
            b_col0 = 2 * t
            b_col1 = 2 * t + 1

            # For A0: A entries are A[i, j] for i in {0,1}, j in {0,1}
            # For A1: A entries are A[i, j+2] for i in {0,1}, j in {0,1}
            # So A_col_offset = 2 * a_block

            a_col_offset = 2 * a_block

            # Strassen 7 terms
            # Term 1: M1 = (a00+a11) * (b00+b11)
            # A part: a00 (i=0,j=0) + a11 (i=1,j=1)
            # B part: b00 (row=b_row0, col=b_col0) + b11 (row=b_row1, col=b_col1)
            # C part: C00 +, C11 +
            # C00: i'=0, k'=2*t
            # C11: i'=1, k'=2*t+1

            # For each term, we add contributions to U, V, W.
            # U[i*m+j, col] where j = a_col_offset + j_sub
            # V[b_row*p + b_col, col]
            # W[k'*n + i', col] where i' is the row of C

            # M1
            U[0 * m + a_col_offset + 0][col] += 1  # a00
            U[1 * m + a_col_offset + 1][col] += 1  # a11
            V[b_row0 * p + b_col0][col] += 1  # b00
            V[b_row1 * p + b_col1][col] += 1  # b11
            W[(2*t) * n + 0][col] += 1  # C[0, 2*t]
            W[(2*t+1) * n + 1][col] += 1  # C[1, 2*t+1]
            col += 1

            # M2 = (a10+a11) * b00
            # C10 +, C11 -
            U[1 * m + a_col_offset + 0][col] += 1  # a10
            U[1 * m + a_col_offset + 1][col] += 1  # a11
            V[b_row0 * p + b_col0][col] += 1  # b00
            W[(2*t) * n + 1][col] += 1  # C[1, 2*t]
            W[(2*t+1) * n + 1][col] += -1  # C[1, 2*t+1]
            col += 1

            # M3 = a00 * (b01-b11)
            # C01 +, C11 +
            U[0 * m + a_col_offset + 0][col] += 1  # a00
            V[b_row0 * p + b_col1][col] += 1  # b01
            V[b_row1 * p + b_col1][col] += -1  # b11
            W[(2*t+1) * n + 0][col] += 1  # C[0, 2*t+1]
            W[(2*t+1) * n + 1][col] += 1  # C[1, 2*t+1]
            col += 1

            # M4 = a11 * (b10-b00)
            # C00 +, C10 +
            U[1 * m + a_col_offset + 1][col] += 1  # a11
            V[b_row1 * p + b_col0][col] += 1  # b10
            V[b_row0 * p + b_col0][col] += -1  # b00
            W[(2*t) * n + 0][col] += 1  # C[0, 2*t]
            W[(2*t) * n + 1][col] += 1  # C[1, 2*t]
            col += 1

            # M5 = (a00+a01) * b11
            # C00 -, C01 +
            U[0 * m + a_col_offset + 0][col] += 1  # a00
            U[0 * m + a_col_offset + 1][col] += 1  # a01
            V[b_row1 * p + b_col1][col] += 1  # b11
            W[(2*t) * n + 0][col] += -1  # C[0, 2*t]
            W[(2*t+1) * n + 0][col] += 1  # C[0, 2*t+1]
            col += 1

            # M6 = (a10-a00) * (b00+b01)
            # C11 +
            U[1 * m + a_col_offset + 0][col] += 1  # a10
            U[0 * m + a_col_offset + 0][col] += -1  # a00
            V[b_row0 * p + b_col0][col] += 1  # b00
            V[b_row0 * p + b_col1][col] += 1  # b01
            W[(2*t+1) * n + 1][col] += 1  # C[1, 2*t+1]
            col += 1

            # M7 = (a01-a11) * (b10+b11)
            # C00 +
            U[0 * m + a_col_offset + 1][col] += 1  # a01
            U[1 * m + a_col_offset + 1][col] += -1  # a11
            V[b_row1 * p + b_col0][col] += 1  # b10
            V[b_row1 * p + b_col1][col] += 1  # b11
            W[(2*t) * n + 0][col] += 1  # C[0, 2*t]
            col += 1

    return {"U": U, "V": V, "W": W}
