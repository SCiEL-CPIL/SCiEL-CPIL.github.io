"""Matrix multiplication 2,4,8 with Strassen-based rank-56 decomposition."""

def construct():
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0.0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0.0] * rank for _ in range(p * n)]  # 16 x 56

    # A is 2x4: A[i,j] -> U index i*m + j = i*4 + j
    # B is 4x8: B[i,j] -> V index i*p + j = i*8 + j
    # Z is 2x8: Z[i,j] -> W index j*n + i = j*2 + i  (trace convention W[k*n+i,r])

    # Split A into A1 (cols 0,1) and A2 (cols 2,3)
    # Split B into B1 (rows 0,1) and B2 (rows 2,3)
    # Z = A1*B1 + A2*B2
    #
    # For each block pair (A_block, B_block), we have a 2x2 * 2x8 product.
    # Split B_block into 4 sub-blocks of 2x2: B_block[q] for q=0..3
    # For each sub-block, apply Strassen's 7-product method.
    # Total: 2 (A blocks) * 4 (B sub-blocks) * 7 (Strassen) = 56 rank-1 tensors.

    # Strassen formulas for 2x2 * 2x2:
    # M1 = (A[1,1] - A[0,1]) * (B[0,0] + B[1,0])
    # M2 = (A[0,0] + A[0,1]) * B[1,1]
    # M3 = A[0,0] * (B[0,1] - B[1,1])
    # M4 = A[1,1] * (B[1,0] - B[0,0])
    # M5 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # M6 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    # M7 = (A[0,0] - A[0,1]) * (B[0,0] + B[0,1])
    #
    # C[0,0] = M5 + M4 - M1 - M7
    # C[0,1] = M3 + M5
    # C[1,0] = M1 + M2
    # C[1,1] = M5 - M3 + M6 - M2

    col = 0
    for a_block in range(2):  # 0: A1 (cols 0,1), 1: A2 (cols 2,3)
        for b_sub in range(4):  # 0..3: sub-blocks of 2x2 within the 2x8 B block
            # A_block indices in U:
            # a_block=0: A[0,0]=idx0, A[0,1]=idx1, A[1,0]=idx4, A[1,1]=idx5
            # a_block=1: A[0,2]=idx2, A[0,3]=idx3, A[1,2]=idx6, A[1,3]=idx7
            a00 = a_block * 2 + 0  # A[0, 2*a_block]
            a01 = a_block * 2 + 1  # A[0, 2*a_block+1]
            a10 = 4 + a_block * 2 + 0  # A[1, 2*a_block]
            a11 = 4 + a_block * 2 + 1  # A[1, 2*a_block+1]

            # B_sub indices in V:
            # b_sub=0: rows 0,1 cols 0,1 -> B[0,0]=idx0, B[0,1]=idx1, B[1,0]=idx8, B[1,1]=idx9
            # b_sub=1: rows 0,1 cols 2,3 -> B[0,2]=idx2, B[0,3]=idx3, B[1,2]=idx10, B[1,3]=idx11
            # b_sub=2: rows 0,1 cols 4,5 -> B[0,4]=idx4, B[0,5]=idx5, B[1,4]=idx12, B[1,5]=idx13
            # b_sub=3: rows 0,1 cols 6,7 -> B[0,6]=idx6, B[0,7]=idx7, B[1,6]=idx14, B[1,7]=idx15
            # For a_block=1, rows are 2,3:
            # b_sub=0: B[2,0]=idx16, B[2,1]=idx17, B[3,0]=idx24, B[3,1]=idx25
            # b_sub=1: B[2,2]=idx18, B[2,3]=idx19, B[3,2]=idx26, B[3,3]=idx27
            # b_sub=2: B[2,4]=idx20, B[2,5]=idx21, B[3,4]=idx28, B[3,5]=idx29
            # b_sub=3: B[2,6]=idx22, B[2,7]=idx23, B[3,6]=idx30, B[3,7]=idx31

            b_row0 = a_block * 2  # 0 or 2
            b_row1 = a_block * 2 + 1  # 1 or 3
            b_col0 = b_sub * 2
            b_col1 = b_sub * 2 + 1

            b00 = b_row0 * 8 + b_col0  # B[b_row0, b_col0]
            b01 = b_row0 * 8 + b_col1  # B[b_row0, b_col1]
            b10 = b_row1 * 8 + b_col0  # B[b_row1, b_col0]
            b11 = b_row1 * 8 + b_col1  # B[b_row1, b_col1]

            # W indices: Z[0, b_col0]=idx b_col0*2+0, Z[0, b_col1]=idx b_col1*2+0
            #             Z[1, b_col0]=idx b_col0*2+1, Z[1, b_col1]=idx b_col1*2+1
            w00 = b_col0 * 2 + 0  # Z[0, b_col0]
            w01 = b_col1 * 2 + 0  # Z[0, b_col1]
            w10 = b_col0 * 2 + 1  # Z[1, b_col0]
            w11 = b_col1 * 2 + 1  # Z[1, b_col1]

            # M1 = (A[1,1] - A[0,1]) * (B[0,0] + B[1,0])
            U[a11][col] += 1.0
            U[a01][col] += -1.0
            V[b00][col] += 1.0
            V[b10][col] += 1.0
            W[w00][col] += -1.0
            W[w10][col] += 1.0
            col += 1

            # M2 = (A[0,0] + A[0,1]) * B[1,1]
            U[a00][col] += 1.0
            U[a01][col] += 1.0
            V[b11][col] += 1.0
            W[w10][col] += 1.0
            W[w11][col] += -1.0
            col += 1

            # M3 = A[0,0] * (B[0,1] - B[1,1])
            U[a00][col] += 1.0
            V[b01][col] += 1.0
            V[b11][col] += -1.0
            W[w01][col] += 1.0
            W[w11][col] += -1.0
            col += 1

            # M4 = A[1,1] * (B[1,0] - B[0,0])
            U[a11][col] += 1.0
            V[b10][col] += 1.0
            V[b00][col] += -1.0
            W[w00][col] += 1.0
            col += 1

            # M5 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
            U[a00][col] += 1.0
            U[a11][col] += 1.0
            V[b00][col] += 1.0
            V[b11][col] += 1.0
            W[w00][col] += 1.0
            W[w01][col] += 1.0
            W[w10][col] += 1.0
            W[w11][col] += 1.0
            col += 1

            # M6 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
            U[a01][col] += 1.0
            U[a11][col] += -1.0
            V[b10][col] += 1.0
            V[b11][col] += 1.0
            W[w11][col] += 1.0
            col += 1

            # M7 = (A[0,0] - A[0,1]) * (B[0,0] + B[0,1])
            U[a00][col] += 1.0
            U[a01][col] += -1.0
            V[b00][col] += 1.0
            V[b01][col] += 1.0
            W[w00][col] += -1.0
            col += 1

    return {"U": U, "V": V, "W": W}
