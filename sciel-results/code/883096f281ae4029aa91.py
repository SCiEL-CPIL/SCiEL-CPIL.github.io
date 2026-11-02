"""Tensor decomposition for 2x3x8 matrix multiplication using Strassen + trivial split."""

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 8
    rank = 44  # 28 (Strassen for 2x2x8) + 16 (trivial for 2x1x8)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Part 1: 2x2x8 using Strassen (rank 7 per 2x2x2 block, 4 blocks)
    # Strassen 2x2x2: M1..M7
    # M1 = (A11)(B11+B22)
    # M2 = (A11+A22)(B22)
    # M3 = (A21)(B11)
    # M4 = (A22)(B21-B11)
    # M5 = (A11+A12)(B22+B21)
    # M6 = (A12-A22)(B11+B12)
    # M7 = (A21-A11)(B12+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6

    # For 2x2x8, we have 4 independent 2x2x2 blocks (columns 0-1, 2-3, 4-5, 6-7 of B)
    # Each block uses rank 7, total 28

    # Strassen multipliers for one 2x2x2 block:
    # For block b (columns 2b, 2b+1 of B):
    # U indices for A: A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1) -> linear indices 0,1,2,3
    # V indices for B: B11=(0,2b), B12=(0,2b+1), B21=(1,2b), B22=(1,2b+1)
    # W indices for C: C11=(0,2b), C12=(0,2b+1), C21=(1,2b), C22=(1,2b+1)

    col = 0
    for b in range(4):
        # B column indices for this block
        bcol0 = 2 * b
        bcol1 = 2 * b + 1

        # M1 = A11 * (B11 + B22)
        U[0][col] = 1
        V[0 * p + bcol0][col] = 1
        V[1 * p + bcol1][col] = 1
        W[bcol0 * n + 0][col] = 1
        W[bcol1 * n + 1][col] = 1
        col += 1

        # M2 = (A11 + A22) * B22
        U[0][col] = 1
        U[3][col] = 1
        V[1 * p + bcol1][col] = 1
        W[bcol1 * n + 1][col] = -1
        W[bcol0 * n + 0][col] = 1
        col += 1

        # M3 = A21 * B11
        U[2][col] = 1
        V[0 * p + bcol0][col] = 1
        W[bcol0 * n + 0][col] = 1
        W[bcol1 * n + 1][col] = 1
        col += 1

        # M4 = A22 * (B21 - B11)
        U[3][col] = 1
        V[1 * p + bcol0][col] = 1
        V[0 * p + bcol0][col] = -1
        W[bcol0 * n + 0][col] = 1
        W[bcol1 * n + 1][col] = -1
        col += 1

        # M5 = (A11 + A12) * (B22 + B21)
        U[0][col] = 1
        U[1][col] = 1
        V[1 * p + bcol1][col] = 1
        V[1 * p + bcol0][col] = 1
        W[bcol0 * n + 0][col] = -1
        W[bcol1 * n + 1][col] = 1
        col += 1

        # M6 = (A12 - A22) * (B11 + B12)
        U[1][col] = 1
        U[3][col] = -1
        V[0 * p + bcol0][col] = 1
        V[0 * p + bcol1][col] = 1
        W[bcol1 * n + 1][col] = 1
        col += 1

        # M7 = (A21 - A11) * (B12 + B22)
        U[2][col] = 1
        U[0][col] = -1
        V[0 * p + bcol1][col] = 1
        V[1 * p + bcol1][col] = 1
        W[bcol0 * n + 0][col] = 1
        col += 1

    # Part 2: 2x1x8 using trivial decomposition (rank 16)
    # A2 is 2x1 (column 2 of A), B2 is 1x8 (row 2 of B)
    # C2 = A2 * B2, so C2[i][k] = A2[i][0] * B2[0][k]
    # For each i in {0,1} and k in {0..7}:
    U[0][col] = 1
    V[2 * p + 0][col] = 1
    W[0 * n + 0][col] = 1
    col += 1
    U[1][col] = 1
    V[2 * p + 0][col] = 1
    W[0 * n + 1][col] = 1
    col += 1
    U[0][col] = 1
    V[2 * p + 1][col] = 1
    W[1 * n + 0][col] = 1
    col += 1
    U[1][col] = 1
    V[2 * p + 1][col] = 1
    W[1 * n + 1][col] = 1
    col += 1
    U[0][col] = 1
    V[2 * p + 2][col] = 1
    W[2 * n + 0][col] = 1
    col += 1
    U[1][col] = 1
    V[2 * p + 2][col] = 1
    W[2 * n + 1][col] = 1
    col += 1
    U[0][col] = 1
    V[2 * p + 3][col] = 1
    W[3 * n + 0][col] = 1
    col += 1
    U[1][col] = 1
    V[2 * p + 3][col] = 1
    W[3 * n + 1][col] = 1
    col += 1
    U[0][col] = 1
    V[2 * p + 4][col] = 1
    W[4 * n + 0][col] = 1
    col += 1
    U[1][col] = 1
    V[2 * p + 4][col] = 1
    W[4 * n + 1][col] = 1
    col += 1
    U[0][col] = 1
    V[2 * p + 5][col] = 1
    W[5 * n + 0][col] = 1
    col += 1
    U[1][col] = 1
    V[2 * p + 5][col] = 1
    W[5 * n + 1][col] = 1
    col += 1
    U[0][col] = 1
    V[2 * p + 6][col] = 1
    W[6 * n + 0][col] = 1
    col += 1
    U[1][col] = 1
    V[2 * p + 6][col] = 1
    W[6 * n + 1][col] = 1
    col += 1
    U[0][col] = 1
    V[2 * p + 7][col] = 1
    W[7 * n + 0][col] = 1
    col += 1
    U[1][col] = 1
    V[2 * p + 7][col] = 1
    W[7 * n + 1][col] = 1
    col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
