"""Strassen-style rank-7 decomposition for 2x2 matrix multiplication.

The public task contract asks for a rank-minimal exact tensor decomposition
of multiplying a 2 by 2 matrix by a 2 by 2 matrix.  The known rank-minimal
decomposition over the reals (and over rationals) for this tensor is the
rank-7 Strassen decomposition, improving on the rank-8 schoolbook form.
"""

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen products (A indices are ij, B indices are jk, C indices are ik).
    #
    # m0 = A00 * (B01 - B11)
    # m1 = (A10 + A11) * B00
    # m2 = (A00 - A10) * (B00 + B01)
    # m3 = A11 * (B10 - B00)
    # m4 = (A00 + A01) * B11
    # m5 = (A01 - A11) * (B10 + B11)
    # m6 = (A10 - A00) * (B01 + B10)
    #
    # C00 = m0 + m2 + m4 + m5
    # C01 = m4 + m6
    # C10 = m1 + m3
    # C11 = m0 + m1 + m2 + m3

    # term 0: m0 = A00 * (B01 - B11)
    U[0][0] = 1
    V[1][0] = 1
    V[3][0] = -1
    W[0][0] = 1
    W[3][0] = 1

    # term 1: m1 = (A10 + A11) * B00
    U[2][1] = 1
    U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    W[3][1] = 1

    # term 2: m2 = (A00 - A10) * (B00 + B01)
    U[0][2] = 1
    U[2][2] = -1
    V[0][2] = 1
    V[1][2] = 1
    W[0][2] = 1
    W[3][2] = 1

    # term 3: m3 = A11 * (B10 - B00)
    U[3][3] = 1
    V[2][3] = 1
    V[0][3] = -1
    W[1][3] = 1
    W[3][3] = 1

    # term 4: m4 = (A00 + A01) * B11
    U[0][4] = 1
    U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1
    W[2][4] = 1

    # term 5: m5 = (A01 - A11) * (B10 + B11)
    U[1][5] = 1
    U[3][5] = -1
    V[2][5] = 1
    V[3][5] = 1
    W[0][5] = 1

    # term 6: m6 = (A10 - A00) * (B01 + B10)
    U[2][6] = 1
    U[0][6] = -1
    V[1][6] = 1
    V[2][6] = 1
    W[2][6] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
