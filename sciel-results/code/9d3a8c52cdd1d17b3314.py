"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-55 decomposition for 2x3 x 3x10 matrix multiplication.
    Strategy:
      - Split the middle dimension (3) into a 2-wide part and a 1-wide part.
      - For the 2-wide part: apply Strassen (rank 7) to each of 5 disjoint 2x2x2 blocks.
        That gives 5 * 7 = 35 rank terms.
      - For the 1-wide part: use schoolbook for the remaining 2x1 x 1x10 subproblem.
        That gives 2 * 1 * 10 = 20 rank terms.
    Total rank = 55.
    """
    n, m, p = DIMENSIONS  # 2, 3, 10
    rank = 55
    U = [[0] * rank for _ in range(n * m)]   # 6 x 55
    V = [[0] * rank for _ in range(m * p)]   # 30 x 55
    W = [[0] * rank for _ in range(p * n)]   # 20 x 55

    col = 0
    # Strassen blocks for the first 2 columns of A, first 2 rows of B
    for b in range(5):
        # B columns: 2*b, 2*b+1
        b0 = 2 * b
        b1 = 2 * b + 1

        # Strassen's 7 products for 2x2 matrices
        # U indices: A rows 0,1; cols 0,1 -> U[0], U[1], U[3], U[4]
        # V indices: B rows 0,1; cols b0, b1 -> V[b0], V[b1], V[10+b0], V[10+b1]
        # W indices: C rows 0,1; cols b0, b1 -> W[b0*2+0], W[b0*2+1], W[b1*2+0], W[b1*2+1]

        # M1 = (A00 + A10)(B00 + B11)
        U[0][col] += 1
        U[3][col] += 1
        V[b0][col] += 1
        V[10 + b1][col] += 1
        W[b0 * 2 + 0][col] += 1
        W[b1 * 2 + 1][col] += 1
        col += 1

        # M2 = A10(B00)
        U[3][col] += 1
        V[b0][col] += 1
        W[b0 * 2 + 1][col] += 1
        W[b1 * 2 + 1][col] -= 1
        col += 1

        # M3 = A00(B01 - B11)
        U[0][col] += 1
        V[b1][col] += 1
        V[10 + b1][col] -= 1
        W[b1 * 2 + 0][col] += 1
        W[b1 * 2 + 1][col] += 1
        col += 1

        # M4 = A10(B10)
        U[3][col] += 1
        V[b0][col] -= 1
        V[10 + b0][col] += 1
        W[b0 * 2 + 0][col] += 1
        W[b0 * 2 + 1][col] += 1
        col += 1

        # M5 = (A00 + A01)(B11)
        U[0][col] += 1
        U[1][col] += 1
        V[10 + b1][col] += 1
        W[b0 * 2 + 0][col] -= 1
        W[b1 * 2 + 0][col] += 1
        col += 1

        # M6 = (A01 - A00)(B00 + B01)
        U[0][col] -= 1
        U[3][col] += 1
        V[b0][col] += 1
        V[b1][col] += 1
        W[b1 * 2 + 1][col] += 1
        col += 1

        # M7 = (A01 + A10)(B10 - B11)
        U[1][col] += 1
        U[3][col] -= 1
        V[10 + b0][col] += 1
        V[10 + b1][col] += 1
        W[b0 * 2 + 0][col] += 1
        col += 1

    # Schoolbook for the remaining 1 column (col 2) of A
    for i in range(2):
        for k in range(10):
            U[i * 3 + 2][col] += 1
            V[2 * 10 + k][col] += 1
            W[k * 2 + i][col] += 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
