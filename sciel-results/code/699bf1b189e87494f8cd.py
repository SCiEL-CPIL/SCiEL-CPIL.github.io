"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition of the 4x4 x 4x4 matrix multiplication tensor
    via recursive Strassen: 7 products of 2x2 matrices, each with rank 7.
    """
    n, m, p = DIMENSIONS  # all 4

    # Strassen 2x2 decomposition
    # U2: (n*m = 4) x 7
    # V2: (m*p = 4) x 7
    # W2: (p*n = 4) x 7
    # Indexing: U[i*m+j], V[j*p+k], W[k*n+i] with m=p=n=2
    # For 2x2: U indices 0..3 map to (i,j) with i in 0..1, j in 0..1
    #           V indices 0..3 map to (j,k) with j in 0..1, k in 0..1
    #           W indices 0..3 map to (k,i) with k in 0..1, i in 0..1

    # Strassen 7 terms for 2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # U2[i*2+j, t] for t=0..6
    U2 = [[0]*7 for _ in range(4)]
    # V2[j*2+k, t]
    V2 = [[0]*7 for _ in range(4)]
    # W2[k*2+i, t]
    W2 = [[0]*7 for _ in range(4)]

    # M1 = (A11+A22)*(B11+B22)
    U2[0][0] = 1  # A11
    U2[3][0] = 1  # A22
    V2[0][0] = 1  # B11
    V2[3][0] = 1  # B22
    W2[0][0] = 1  # C11 += M1
    W2[3][0] = 1  # C22 += M1

    # M2 = (A21+A22)*B11
    U2[2][1] = 1  # A21
    U2[3][1] = 1  # A22
    V2[0][1] = 1  # B11
    W2[2][1] = 1  # C21 += M2
    W2[3][1] = -1 # C22 -= M2

    # M3 = A11*(B12-B22)
    U2[0][2] = 1  # A11
    V2[1][2] = 1  # B12
    V2[3][2] = -1 # B22
    W2[1][2] = 1  # C12 += M3
    W2[3][2] = 1  # C22 += M3

    # M4 = A22*(B21-B11)
    U2[3][3] = 1  # A22
    V2[2][3] = 1  # B21
    V2[0][3] = -1 # B11
    W2[0][3] = 1  # C11 += M4
    W2[2][3] = 1  # C21 += M4

    # M5 = (A11+A12)*B22
    U2[0][4] = 1  # A11
    U2[1][4] = 1  # A12
    V2[3][4] = 1  # B22
    W2[0][4] = -1 # C11 -= M5
    W2[1][4] = 1  # C12 += M5

    # M6 = (A21-A11)*(B11+B12)
    U2[2][5] = 1  # A21
    U2[0][5] = -1 # A11
    V2[0][5] = 1  # B11
    V2[1][5] = 1  # B12
    W2[3][5] = 1  # C22 += M6

    # M7 = (A12-A22)*(B21+B22)
    U2[1][6] = 1  # A12
    U2[3][6] = -1 # A22
    V2[2][6] = 1  # B21
    V2[3][6] = 1  # B22
    W2[0][6] = 1  # C11 += M7

    # Now build 4x4 by Kronecker product of Strassen with itself
    # The 4x4 matrix is a 2x2 block matrix of 2x2 blocks.
    # A = [[A11, A12], [A21, A22]] where each Aij is 2x2
    # Similarly B and C.
    # We apply Strassen at the block level:
    # M1_block = (A11+A22)*(B11+B22)
    # M2_block = (A21+A22)*B11
    # etc.
    # Each M*_block is a product of 2x2 matrices, which we decompose with Strassen 2x2.

    # The full decomposition has rank 7*7 = 49.
    # For each block-level Strassen term t (0..6), we have a product of two 2x2 matrices.
    # The 2x2 matrices are linear combinations of the 4x4 entries.

    # Let's define the block-level Strassen:
    # For term t, U_block[t] gives coefficients on A-blocks, V_block[t] on B-blocks.
    # U_block: 4x4 blocks A11,A12,A21,A22 -> each is 2x2
    # Actually, let's think in terms of the 4x4 entries directly.

    # A 4x4 matrix A has entries A[i][j] for i,j in 0..3.
    # Block A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]

    # For block Strassen term t, the input to the product is:
    # X_t = sum over blocks b of U_block[t][b] * A_b  (this is a 2x2 matrix)
    # Y_t = sum over blocks b of V_block[t][b] * B_b  (this is a 2x2 matrix)
    # Product P_t = X_t * Y_t (2x2 result)
    # Then C_block_c += W_block[t][c] * P_t

    # Block-level Strassen coefficients (same pattern as 2x2 Strassen):
    # U_block[t][b] where b in {0=A11, 1=A12, 2=A21, 3=A22}
    U_block = [
        [1, 0, 0, 1],   # M1: A11+A22
        [0, 0, 1, 1],   # M2: A21+A22
        [1, 0, 0, 0],   # M3: A11
        [0, 0, 0, 1],   # M4: A22
        [1, 1, 0, 0],   # M5: A11+A12
        [-1, 0, 1, 0],  # M6: A21-A11
        [0, 1, 0, -1],  # M7: A12-A22
    ]
    V_block = [
        [1, 0, 0, 1],   # M1: B11+B22
        [1, 0, 0, 0],   # M2: B11
        [0, 1, 0, -1],  # M3: B12-B22
        [0, 0, 1, -1],  # M4: B21-B11
        [0, 0, 0, 1],   # M5: B22
        [1, 1, 0, 0],   # M6: B11+B12
        [0, 0, 1, 1],   # M7: B21+B22
    ]
    W_block = [
        [1, 0, 0, 1],   # M1: C11+=, C22+=
        [0, 0, 1, -1],  # M2: C21+=, C22-=
        [0, 1, 0, 1],   # M3: C12+=, C22+=
        [1, 0, 1, 0],   # M4: C11+=, C21+=
        [-1, 1, 0, 0],  # M5: C11-=, C12+=
        [0, 0, 0, 1],   # M6: C22+=
        [1, 0, 0, 0],   # M7: C11+=
    ]

    # Now, for each block Strassen term t (0..6), we need to decompose the 2x2 product
    # X_t * Y_t using the 2x2 Strassen decomposition (U2, V2, W2).

    # The full U, V, W will have 49 columns.
    # Column index = t * 7 + s where t is block Strassen term, s is inner Strassen term.

    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]

    # For block term t and inner term s:
    # X_t[i_inner][j_inner] = sum_b U_block[t][b] * A_block_b[i_inner][j_inner]
    # Y_t[j_inner][k_inner] = sum_b V_block[t][b] * B_block_b[j_inner][k_inner]

    # A_block_b: b=0 -> A[0:2,0:2], b=1 -> A[0:2,2:4], b=2 -> A[2:4,0:2], b=3 -> A[2:4,2:4]
    # Block b has rows (2*b_row, 2*b_row+1) and cols (2*b_col, 2*b_col+1)
    # b=0: rows 0,1 cols 0,1
    # b=1: rows 0,1 cols 2,3
    # b=2: rows 2,3 cols 0,1
    # b=3: rows 2,3 cols 2,3

    # For the U matrix: U[row_idx, col_idx]
    # row_idx = i*4 + j where i is row in 4x4, j is col in 4x4
    # The inner Strassen U2 has entries U2[i2*2+j2, s]
    # For inner term s, U2[i2*2+j2, s] multiplies A_entry[i2, j2] within the 2x2 block

    # So for block term t, inner term s:
    # The coefficient on A[i][j] is:
    #   sum_b U_block[t][b] * U2[(i%2)*2 + (j%2), s]  if (i,j) is in block b
    # Block b for entry (i,j): b_row = i//2, b_col = j//2, b = b_row*2 + b_col

    for t in range(7):
        for s in range(7):
            col_idx = t * 7 + s
            for i in range(4):
                for j in range(4):
                    b_row = i // 2
                    b_col = j // 2
                    b = b_row * 2 + b_col
                    coeff = U_block[t][b] * U2[(i % 2) * 2 + (j % 2)][s]
                    U[i * 4 + j][col_idx] += coeff

            for j in range(4):
                for k in range(4):
                    b_row = j // 2
                    b_col = k // 2
                    b = b_row * 2 + b_col
                    coeff = V_block[t][b] * V2[(j % 2) * 2 + (k % 2)][s]
                    V[j * 4 + k][col_idx] += coeff

            for k in range(4):
                for i in range(4):
                    b_row = k // 2
                    b_col = i // 2
                    b = b_row * 2 + b_col
                    coeff = W_block[t][b] * W2[(k % 2) * 2 + (i % 2)][s]
                    W[k * 4 + i][col_idx] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
