"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10

    # Block-Kronecker structure:
    # - First 2 columns of B: rank-11 tensor (Strassen for 2x2 + schoolbook for 2x1)
    # - Remaining 8 columns of B: five rank-10 tensors (schoolbook)
    # Total rank: 11 + 5*10 = 61

    rank_total = 61
    U = [[0] * rank_total for _ in range(n * m)]  # (n*m, rank) = (6, 61)
    V = [[0] * rank_total for _ in range(m * p)]  # (m*p, rank) = (30, 61)
    W = [[0] * rank_total for _ in range(p * n)]  # (p*n, rank) = (20, 61)

    col = 0

    # Part 1: First 2 columns of B (k=0, 1)
    # A is 2x3, B1 is 3x2
    # A = [[a00, a01, a02], [a10, a11, a12]]
    # B1 = [[b00, b01], [b10, b11], [b20, b21]]
    # C1 = A*B1

    # Strassen for A1 (2x2) * B11 (2x2):
    # A1 = [[a00, a01], [a10, a11]]
    # B11 = [[b00, b01], [b10, b11]]

    # M1 = (a00+a11)*(b00+b11)
    U[0 * m + 0][col] = 1; U[1 * m + 1][col] = 1
    V[0 * p + 0][col] = 1; V[1 * p + 1][col] = 1
    W[0 * n + 0][col] = 1; W[0 * n + 1][col] = 1; W[1 * n + 0][col] = 1; W[1 * n + 1][col] = 1
    col += 1

    # M2 = (a10+a11)*b00
    U[1 * m + 0][col] = 1; U[1 * m + 1][col] = 1
    V[0 * p + 0][col] = 1
    W[1 * n + 0][col] = 1
    col += 1

    # M3 = a00*(b01-b11)
    U[0 * m + 0][col] = 1
    V[0 * p + 1][col] = 1; V[1 * p + 1][col] = -1
    W[0 * n + 1][col] = 1
    col += 1

    # M4 = a11*(b10-b00)
    U[1 * m + 1][col] = 1
    V[1 * p + 0][col] = 1; V[0 * p + 0][col] = -1
    W[1 * n + 0][col] = 1
    col += 1

    # M5 = (a00+a01)*b11
    U[0 * m + 0][col] = 1; U[0 * m + 1][col] = 1
    V[1 * p + 1][col] = 1
    W[0 * n + 0][col] = -1; W[0 * n + 1][col] = 1
    col += 1

    # M6 = (a10-a00)*(b00+b01)
    U[1 * m + 0][col] = 1; U[0 * m + 0][col] = -1
    V[0 * p + 0][col] = 1; V[0 * p + 1][col] = 1
    W[1 * n + 1][col] = 1
    col += 1

    # M7 = (a01-a11)*(b10+b11)
    U[0 * m + 1][col] = 1; U[1 * m + 1][col] = -1
    V[1 * p + 0][col] = 1; V[1 * p + 1][col] = 1
    W[0 * n + 0][col] = 1; W[1 * n + 1][col] = -1
    col += 1

    # Schoolbook for A2 * B12:
    # a02*b20, a02*b21, a12*b20, a12*b21
    U[0 * m + 2][col] = 1
    V[2 * p + 0][col] = 1
    W[0 * n + 0][col] = 1
    col += 1

    U[0 * m + 2][col] = 1
    V[2 * p + 1][col] = 1
    W[0 * n + 1][col] = 1
    col += 1

    U[1 * m + 2][col] = 1
    V[2 * p + 0][col] = 1
    W[1 * n + 0][col] = 1
    col += 1

    U[1 * m + 2][col] = 1
    V[2 * p + 1][col] = 1
    W[1 * n + 1][col] = 1
    col += 1

    # Part 2: Remaining 8 columns of B (k=2,3,...,9)
    # Schoolbook for A(2x3) * B2(3x8)
    # This gives 2*3*8 = 48 terms, but we group them as five rank-10 tensors
    # Actually, let's just do schoolbook for the remaining 8 columns
    for i in range(n):
        for j in range(m):
            for k in range(2, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
