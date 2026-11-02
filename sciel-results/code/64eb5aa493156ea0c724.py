import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 5
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 8x36
    V = [[0] * rank for _ in range(m * p)]  # 20x36
    W = [[0] * rank for _ in range(p * n)]  # 10x36

    col = 0

    # Helper to add a Strassen block
    # A is 2x2, B is 2x2, C is 2x2
    # A indices: A[row, col] in U
    # B indices: B[row, col] in V
    # C indices: C[row, col] in W
    
    def add_strassen(a_rows, b_rows, c_rows):
        nonlocal col
        # a_rows: list of 4 indices in U for A11, A12, A21, A22
        # b_rows: list of 4 indices in V for B11, B12, B21, B22
        # c_rows: list of 4 indices in W for C11, C12, C21, C22
        
        a11, a12, a21, a22 = a_rows
        b11, b12, b21, b22 = b_rows
        c11, c12, c21, c22 = c_rows

        # m1 = (a11 + a22)(b11 + b22)
        U[a11][col] = 1; U[a22][col] = 1
        V[b11][col] = 1; V[b22][col] = 1
        W[c11][col] = 1; W[c22][col] = 1
        W[c12][col] = 1; W[c21][col] = 1
        # C11 += m1, C22 += m1, C12 += m1, C21 += m1? No.
        # Strassen:
        # C11 = m1 - m4 + m5 - m7
        # C12 = m3 + m4
        # C21 = m2 + m7
        # C22 = m1 - m2 + m3 + m6
        
        # We need to set up the signs correctly in W.
        # W[c11][col] += 1 (from m1)
        # W[c22][col] += 1 (from m1)
        # But m1 contributes to C11, C12, C21, C22?
        # No, m1 is (A11+A22)(B11+B22).
        # It contributes to all 4 entries of C.
        # C11: +1
        # C12: +1
        # C21: +1
        # C22: +1
        
        W[c11][col] = 1
        W[c12][col] = 1
        W[c21][col] = 1
        W[c22][col] = 1
        col += 1

        # m2 = (a21 + a22)b11
        U[a21][col] = 1; U[a22][col] = 1
        V[b11][col] = 1
        # C21 += m2, C22 -= m2
        W[c21][col] = 1
        W[c22][col] = -1
        col += 1

        # m3 = a11(b12 - b22)
        U[a11][col] = 1
        V[b12][col] = 1; V[b22][col] = -1
        # C12 += m3, C22 += m3
        W[c12][col] = 1
        W[c22][col] = 1
        col += 1

        # m4 = a22(b21 - b11)
        U[a22][col] = 1
        V[b21][col] = 1; V[b11][col] = -1
        # C11 -= m4, C12 += m4
        W[c11][col] = -1
        W[c12][col] = 1
        col += 1

        # m5 = (a11 + a12)b22
        U[a11][col] = 1; U[a12][col] = 1
        V[b22][col] = 1
        # C11 += m5, C22 += m5
        W[c11][col] = 1
        W[c22][col] = 1
        col += 1

        # m6 = (a21 - a11)(b11 + b12)
        U[a21][col] = 1; U[a11][col] = -1
        V[b11][col] = 1; V[b12][col] = 1
        # C22 += m6
        W[c22][col] = 1
        col += 1

        # m7 = (a12 - a22)(b21 + b22)
        U[a12][col] = 1; U[a22][col] = -1
        V[b21][col] = 1; V[b22][col] = 1
        # C11 -= m7, C21 += m7
        W[c11][col] = -1
        W[c21][col] = 1
        col += 1

    # Helper to add a schoolbook block (2x2 matrix times 2x1 vector)
    def add_dot(a_rows, b_row, c_rows):
        nonlocal col
        # a_rows: 2 indices in U (rows of A)
        # b_row: 2 indices in V (rows of B vector)
        # c_rows: 2 indices in W (rows of C vector)
        
        # C0 = A00*B0 + A01*B1
        # C1 = A10*B0 + A11*B1
        
        # mult 1: A00 * B0
        U[a_rows[0]][col] = 1
        V[b_row[0]][col] = 1
        W[c_rows[0]][col] = 1
        col += 1
        
        # mult 2: A01 * B1
        U[a_rows[1]][col] = 1
        V[b_row[1]][col] = 1
        W[c_rows[0]][col] = 1
        col += 1
        
        # mult 3: A10 * B0
        U[a_rows[2]][col] = 1
        V[b_row[0]][col] = 1
        W[c_rows[1]][col] = 1
        col += 1
        
        # mult 4: A11 * B1
        U[a_rows[3]][col] = 1
        V[b_row[1]][col] = 1
        W[c_rows[1]][col] = 1
        col += 1

    # Indices mapping:
    # U: A is 2x4. Row 0: A00, A01, A02, A03. Row 1: A10, A11, A12, A13.
    # A1 = A[:, 0:2] -> U indices: [0, 1, 2, 3] (A00, A01, A10, A11)
    # A2 = A[:, 2:4] -> U indices: [4, 5, 6, 7] (A02, A03, A12, A13)
    
    a1_indices = [0, 1, 2, 3]
    a2_indices = [4, 5, 6, 7]

    # V: B is 4x5.
    # Row 0: B00, B01, B02, B03, B04
    # Row 1: B10, B11, B12, B13, B14
    # Row 2: B20, B21, B22, B23, B24
    # Row 3: B30, B31, B32, B33, B34
    
    # B1 = B[0:2, 0:2] -> V indices: [0, 1, 2, 3]
    # B2 = B[2:4, 0:2] -> V indices: [10, 11, 12, 13]
    # B3 = B[0:2, 2:4] -> V indices: [2, 3, 4, 5]
    # B4 = B[2:4, 2:4] -> V indices: [12, 13, 14, 15]
    
    # b1 = B[0:2, 4] -> V indices: [4, 9]
    # b2 = B[2:4, 4] -> V indices: [14, 19]

    b1_indices = [0, 1, 2, 3]
    b2_indices = [10, 11, 12, 13]
    b3_indices = [2, 3, 4, 5]
    b4_indices = [12, 13, 14, 15]
    
    vb1_indices = [4, 9]
    vb2_indices = [14, 19]

    # W: C is 2x5. W[k*n + i, r]. n=2.
    # C00 -> W[0], C10 -> W[1]
    # C01 -> W[2], C11 -> W[3]
    # C02 -> W[4], C12 -> W[5]
    # C03 -> W[6], C13 -> W[7]
    # C04 -> W[8], C14 -> W[9]
    
    # C01 block (cols 0,1): W indices [0, 2, 1, 3] (C00, C01, C10, C11)
    # C23 block (cols 2,3): W indices [4, 6, 5, 7] (C02, C03, C12, C13)
    
    w01_indices = [0, 2, 1, 3]
    w23_indices = [4, 6, 5, 7]
    
    # c block (col 4): W indices [8, 9] (C04, C14)
    wc_indices = [8, 9]

    # Execute decomposition
    add_strassen(a1_indices, b1_indices, w01_indices)
    add_strassen(a2_indices, b2_indices, w01_indices)
    add_strassen(a1_indices, b3_indices, w23_indices)
    add_strassen(a2_indices, b4_indices, w23_indices)
    
    # For the last column, we need to sum the results of A1*b1 and A2*b2 into the same W entries.
    # add_dot adds 1 to W. Since we call it twice, it will add 1+1=2? No.
    # The decomposition is linear. We need W to sum the contributions.
    # So we just call add_dot twice. The W entries will accumulate.
    
    add_dot(a1_indices, vb1_indices, wc_indices)
    add_dot(a2_indices, vb2_indices, wc_indices)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
