"""Tensor decomposition for 3x3 matrix multiplication."""

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 3
    rank = 22
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # Index conventions:
    # U row: i*m + j  (i = A row, j = A col)
    # V row: j*p + k  (j = B row, k = B col)
    # W row: k*n + i  (k = C col, i = C row)  [trace convention W[k*n+i,r]]

    # Strassen's 7 products for the top-left 2x2 block:
    # M1 = (A00 - A10)(B01 + B11)
    # M2 = (A00 + A01)(B10)
    # M3 = (A10)(B00 - B10)
    # M4 = (A11)(B01 - B11)
    # M5 = (A00 + A11)(B00)
    # M6 = (A01 - A11)(B10 + B11)
    # M7 = (A00 - A10)(B00 + B10)

    # Then 15 direct monomial terms for the remaining products.

    # Term 0: M1 = (A00 - A10)(B01 + B11) -> C01 += 1, C10 -= 1, C11 += 1
    r = 0
    U[0][r] = 1; U[3][r] = -1
    V[1][r] = 1; V[4][r] = 1
    W[1*3+0][r] = 1  # C row 0, col 1
    W[0*3+1][r] = -1 # C row 1, col 0
    W[1*3+1][r] = 1  # C row 1, col 1

    # Term 1: M2 = (A00 + A01)(B10) -> C00 += 1, C01 += 1
    r = 1
    U[0][r] = 1; U[1][r] = 1
    V[3][r] = 1
    W[0*3+0][r] = 1
    W[1*3+0][r] = 1

    # Term 2: M3 = (A10)(B00 - B10) -> C10 += 1, C11 -= 1
    r = 2
    U[3][r] = 1
    V[0][r] = 1; V[3][r] = -1
    W[0*3+1][r] = 1
    W[1*3+1][r] = -1

    # Term 3: M4 = (A11)(B01 - B11) -> C11 += 1, C12 -= 1
    r = 3
    U[4][r] = 1
    V[1][r] = 1; V[4][r] = -1
    W[1*3+1][r] = 1
    W[2*3+1][r] = -1

    # Term 4: M5 = (A00 + A11)(B00) -> C00 += 1, C11 += 1
    r = 4
    U[0][r] = 1; U[4][r] = 1
    V[0][r] = 1
    W[0*3+0][r] = 1
    W[1*3+1][r] = 1

    # Term 5: M6 = (A01 - A11)(B10 + B11) -> C01 += 1, C10 -= 1
    r = 5
    U[1][r] = 1; U[4][r] = -1
    V[3][r] = 1; V[4][r] = 1
    W[1*3+0][r] = 1
    W[0*3+1][r] = -1

    # Term 6: M7 = (A00 - A10)(B00 + B10) -> C00 += 1, C10 += 1
    r = 6
    U[0][r] = 1; U[3][r] = -1
    V[0][r] = 1; V[3][r] = 1
    W[0*3+0][r] = 1
    W[0*3+1][r] = 1

    # Now 15 direct terms for the border:
    # C02 = A00*B02 + A01*B12 + A02*B22
    # C12 = A10*B02 + A11*B12 + A12*B22
    # C20 = A20*B00 + A21*B10 + A22*B20
    # C21 = A20*B01 + A21*B11 + A22*B21
    # C22 = A20*B02 + A21*B12 + A22*B22

    # Term 7: A00*B02 -> C02
    r = 7
    U[0][r] = 1
    V[2][r] = 1
    W[2*3+0][r] = 1  # C row 0, col 2

    # Term 8: A01*B12 -> C02
    r = 8
    U[1][r] = 1
    V[5][r] = 1
    W[2*3+0][r] = 1

    # Term 9: A02*B22 -> C02
    r = 9
    U[2][r] = 1
    V[8][r] = 1
    W[2*3+0][r] = 1

    # Term 10: A10*B02 -> C12
    r = 10
    U[3][r] = 1
    V[2][r] = 1
    W[2*3+1][r] = 1  # C row 1, col 2

    # Term 11: A11*B12 -> C12
    r = 11
    U[4][r] = 1
    V[5][r] = 1
    W[2*3+1][r] = 1

    # Term 12: A12*B22 -> C12
    r = 12
    U[5][r] = 1
    V[8][r] = 1
    W[2*3+1][r] = 1

    # Term 13: A20*B00 -> C20
    r = 13
    U[6][r] = 1
    V[0][r] = 1
    W[0*3+2][r] = 1  # C row 2, col 0

    # Term 14: A21*B10 -> C20
    r = 14
    U[7][r] = 1
    V[3][r] = 1
    W[0*3+2][r] = 1

    # Term 15: A22*B20 -> C20
    r = 15
    U[8][r] = 1
    V[6][r] = 1
    W[0*3+2][r] = 1

    # Term 16: A20*B01 -> C21
    r = 16
    U[6][r] = 1
    V[1][r] = 1
    W[1*3+2][r] = 1  # C row 2, col 1

    # Term 17: A21*B11 -> C21
    r = 17
    U[7][r] = 1
    V[4][r] = 1
    W[1*3+2][r] = 1

    # Term 18: A22*B21 -> C21
    r = 18
    U[8][r] = 1
    V[7][r] = 1
    W[1*3+2][r] = 1

    # Term 19: A20*B02 -> C22
    r = 19
    U[6][r] = 1
    V[2][r] = 1
    W[2*3+2][r] = 1  # C row 2, col 2

    # Term 20: A21*B12 -> C22
    r = 20
    U[7][r] = 1
    V[5][r] = 1
    W[2*3+2][r] = 1

    # Term 21: A22*B22 -> C22
    r = 21
    U[8][r] = 1
    V[8][r] = 1
    W[2*3+2][r] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
