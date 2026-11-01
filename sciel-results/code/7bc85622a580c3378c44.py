"""Strassen rank-7 exact tensor decomposition for 2x2 matrix multiplication."""

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """Return a rank-7 exact decomposition of the 2x2 matrix multiplication tensor.

    Uses Strassen's 7-term algorithm. The output factor W follows the
    official trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS

    # Strassen's 7 intermediate products (with the standard "M" naming):
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b01 - b11)
    # M4 = a11 * (b10 - b00)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a00) * (b00 + b01)
    # M7 = (a01 - a11) * (b10 + b11)
    #
    # Output (C = A * B):
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    rank = 7
    U = [[0] * rank for _ in range(n * m)]   # shape (4, 7), indexed by [i*m+j, r]
    V = [[0] * rank for _ in range(m * p)]   # shape (4, 7), indexed by [j*p+k, r]
    W = [[0] * rank for _ in range(p * n)]   # shape (4, 7), indexed by [k*n+i, r]

    # U rows: a00 (i=0,j=0), a01 (i=0,j=1), a10 (i=1,j=0), a11 (i=1,j=1)
    # V rows: b00 (j=0,k=0), b01 (j=0,k=1), b10 (j=1,k=0), b11 (j=1,k=1)
    # W rows: c00 (k=0,i=0), c10 (k=1,i=0), c01 (k=0,i=1), c11 (k=1,i=1)

    # M1 = (a00 + a11) * (b00 + b11)
    U[0][0] = 1  # a00
    U[3][0] = 1  # a11
    V[0][0] = 1  # b00
    V[3][0] = 1  # b11
    # C00 += M1, C11 += M1
    W[0][0] = 1  # c00
    W[3][0] = 1  # c11

    # M2 = (a10 + a11) * b00
    U[2][1] = 1  # a10
    U[3][1] = 1  # a11
    V[0][1] = 1  # b00
    # C10 += M2
    W[1][1] = 1  # c10

    # M3 = a00 * (b01 - b11)
    U[0][2] = 1  # a00
    V[1][2] = 1  # b01
    V[3][2] = -1  # -b11
    # C01 += M3, C11 += M3
    W[2][2] = 1  # c01
    W[3][2] = 1  # c11

    # M4 = a11 * (b10 - b00)
    U[3][3] = 1  # a11
    V[2][3] = 1  # b10
    V[0][3] = -1  # -b00
    # C00 += M4, C10 += M4
    W[0][3] = 1  # c00
    W[1][3] = 1  # c10

    # M5 = (a00 + a01) * b11
    U[0][4] = 1  # a00
    U[1][4] = 1  # a01
    V[3][4] = 1  # b11
    # C00 -= M5, C01 += M5
    W[0][4] = -1  # c00
    W[2][4] = 1  # c01

    # M6 = (a10 - a00) * (b00 + b01)
    U[2][5] = 1  # a10
    U[0][5] = -1  # -a00
    V[0][5] = 1  # b00
    V[1][5] = 1  # b01
    # C11 += M6
    W[3][5] = 1  # c11

    # M7 = (a01 - a11) * (b10 + b11)
    U[1][6] = 1  # a01
    U[3][6] = -1  # -a11
    V[2][6] = 1  # b10
    V[3][6] = 1  # b11
    # C00 += M7
    W[0][6] = 1  # c00

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
