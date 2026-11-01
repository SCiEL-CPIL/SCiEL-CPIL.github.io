"""Matrix multiplication 2x2 x 2x3 rank-minimal decomposition (rank 11)."""

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    # Strassen (rank 7) for the 2x2x2 part + schoolbook (rank 4) for the 2x2x1 part.
    # A is 2x2 (rows i=0,1; cols j=0,1), B is 2x3 (rows j=0,1; cols k=0,1,2).
    # C = A*B is 2x3 (rows i=0,1; cols k=0,1,2).
    #
    # Indexing:
    #   U rows:  i*m + j  = i*2 + j  -> 4 rows
    #   V rows:  j*p + k  = j*3 + k  -> 6 rows
    #   W rows:  k*n + i  = k*2 + i  -> 6 rows
    #
    # Strassen 7 multiplications for cols 0,1 of B:
    #   M1 = (A00+A11)(B00+B11)
    #   M2 = (A10+A11)B00
    #   M3 = A00(B10+B11)
    #   M4 = A11(B00+B01)
    #   M5 = (A00+A10)B10
    #   M6 = (A01+A11)(B00+B01)  -- wait, let me use the standard Strassen for 2x2.
    #
    # Standard Strassen for C = A*B where A,B are 2x2:
    #   M1 = (A00 + A11)(B00 + B11)
    #   M2 = (A10 + A11) B00
    #   M3 = A00 (B10 + B11)
    #   M4 = A11 (B00 + B01)
    #   M5 = (A00 + A10) B10
    #   M6 = (A01 + A11) (B00 + B01)
    #   M7 = (A10 + A11) (B10 + B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6
    #
    # For the 2x2x1 part (col 2 of B):
    #   M8  = A00 * B02
    #   M9  = A01 * B12
    #   M10 = A10 * B02
    #   M11 = A11 * B12
    #
    # C02 = M8 + M9
    # C12 = M10 + M11

    rank = 11
    n, m, p = 2, 2, 3

    # Initialize
    U = [[0] * rank for _ in range(n * m)]  # 4 x 11
    V = [[0] * rank for _ in range(m * p)]  # 6 x 11
    W = [[0] * rank for _ in range(p * n)]  # 6 x 11

    # Strassen factors (columns 0-6)
    # M1 = (A00 + A11)(B00 + B11)
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[4][0] += 1
    W[0][0] += 1  # C00
    W[3][0] += 1  # C11

    # M2 = (A10 + A11) B00
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1  # C10
    W[3][1] += -1 # C11

    # M3 = A00 (B10 + B11)
    U[0][2] += 1
    V[3][2] += 1; V[4][2] += 1
    W[1][2] += 1  # C01
    W[3][2] += 1  # C11

    # M4 = A11 (B00 + B01)
    U[3][3] += 1
    V[0][3] += 1; V[1][3] += 1
    W[0][3] += 1  # C00
    W[2][3] += 1  # C10

    # M5 = (A00 + A10) B10
    U[0][4] += 1; U[2][4] += 1
    V[3][4] += 1
    W[0][4] += -1 # C00
    W[2][4] += 1  # C10

    # M6 = (A01 + A11) (B00 + B01)
    U[1][5] += 1; U[3][5] += 1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1  # C11

    # M7 = (A10 + A11) (B10 + B11)
    U[2][6] += 1; U[3][6] += 1
    V[3][6] += 1; V[4][6] += 1
    W[0][6] += -1 # C00
    W[1][6] += 1  # C01

    # Schoolbook for col 2 (columns 7-10)
    # M8 = A00 * B02
    U[0][7] += 1
    V[2][7] += 1  # B02 = row 0, col 2 -> j*p+k = 0*3+2 = 2
    W[4][7] += 1  # C02 = k*n+i = 2*2+0 = 4

    # M9 = A01 * B12
    U[1][8] += 1
    V[5][8] += 1  # B12 = row 1, col 2 -> j*p+k = 1*3+2 = 5
    W[4][8] += 1  # C02

    # M10 = A10 * B02
    U[2][9] += 1
    V[2][9] += 1
    W[5][9] += 1  # C12 = k*n+i = 2*2+1 = 5

    # M11 = A11 * B12
    U[3][10] += 1
    V[5][10] += 1
    W[5][10] += 1 # C12

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
