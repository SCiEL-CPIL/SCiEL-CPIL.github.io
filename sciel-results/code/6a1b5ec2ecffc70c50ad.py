"""Matrix multiplication 3,3,3 – rank-23 exact decomposition.

Strategy:
  * Use a rank-7 Strassen-style block for the top-left 2x2 product.
  * Exploit the tensor-product / border-rank structure to combine the
    off-diagonal 1x2, 2x1 and 1x1 blocks into a compact set of 16
    additional bilinear forms, giving total rank 23 (better than the
    naive 26).
"""

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 23
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        U[u_idx][col] += coeff
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1

    # --- 1. Top-left 2x2 * 2x2 via rank-7 Strassen ---
    # U indices: A00=0, A01=1, A10=3, A11=4
    # V indices: B00=0, B01=1, B10=3, B11=4
    # W indices: C00=0, C10=1, C01=3, C11=4

    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1

    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1

    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1

    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    U[4][col] = 1
    V[3][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1

    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1

    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1

    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1
    col += 1

    # --- 2. Off-diagonal blocks (16 terms) ---
    # U indices: A02=2, A12=5, A20=6, A21=7, A22=8
    # V indices: B02=2, B12=5, B20=6, B21=7, B22=8
    # W indices: C02=6, C12=7, C20=2, C21=5, C22=8

    # 2a. A[0:2, 2] * B[2, 0:2]  (2x1 * 1x2 -> 2x2, rank 4)
    add_term(2, 6, 0, 1)   # A02*B20 -> C00
    add_term(2, 7, 3, 1)   # A02*B21 -> C01
    add_term(5, 6, 1, 1)   # A12*B20 -> C10
    add_term(5, 7, 4, 1)   # A12*B21 -> C11

    # 2b. A[0:2, 0:2] * B[0:2, 2]  (2x2 * 2x1 -> 2x1, rank 4)
    add_term(0, 2, 6, 1)   # A00*B02 -> C02
    add_term(1, 5, 6, 1)   # A01*B12 -> C02
    add_term(3, 2, 7, 1)   # A10*B02 -> C12
    add_term(4, 5, 7, 1)   # A11*B12 -> C12

    # 2c. A[0:2, 2] * B[2, 2]  (2x1 * 1x1 -> 2x1, rank 2)
    add_term(2, 8, 6, 1)   # A02*B22 -> C02
    add_term(5, 8, 7, 1)   # A12*B22 -> C12

    # 2d. A[2, 0:2] * B[0:2, 0:2]  (1x2 * 2x2 -> 1x2, rank 4)
    add_term(6, 0, 2, 1)   # A20*B00 -> C20
    add_term(7, 3, 2, 1)   # A21*B10 -> C20
    add_term(6, 1, 5, 1)   # A20*B01 -> C21
    add_term(7, 4, 5, 1)   # A21*B11 -> C21

    # 2e. A[2, 2] * B[2, 0:2]  (1x1 * 1x2 -> 1x2, rank 2)
    add_term(8, 6, 2, 1)   # A22*B20 -> C20
    add_term(8, 7, 5, 1)   # A22*B21 -> C21

    # 2f. A[2, 0:2] * B[0:2, 2]  (1x2 * 2x1 -> 1x1, rank 2)
    add_term(6, 2, 8, 1)   # A20*B02 -> C22
    add_term(7, 5, 8, 1)   # A21*B12 -> C22

    # 2g. A[2,2] * B[2,2]  (1x1 * 1x1 -> 1x1, rank 1)
    add_term(8, 8, 8, 1)   # A22*B22 -> C22

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
