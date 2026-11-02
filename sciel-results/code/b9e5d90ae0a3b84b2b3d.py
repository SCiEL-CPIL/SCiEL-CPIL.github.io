"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10

    # Block-Kronecker structure:
    # - First 2 columns of B: rank-11 tensor (Strassen for 2x2 + schoolbook for 2x1x2)
    # - Remaining 8 columns of B: schoolbook rank 2*3*8 = 48
    # Total rank: 11 + 48 = 59

    U = [[0.0] * 59 for _ in range(n * m)]  # 6 x 59
    V = [[0.0] * 59 for _ in range(m * p)]  # 30 x 59
    W = [[0.0] * 59 for _ in range(p * n)]  # 20 x 59

    col = 0

    # Part 1: First 2 columns (k=0,1) using Strassen + schoolbook
    # A = [[a00, a01, a02], [a10, a11, a12]]
    # B1 = [[b00, b01], [b10, b11], [b20, b21]]
    # C1 = A*B1 (2x2)

    # Strassen for A1 (2x2) * B11 (2x2):
    # A1 = [[a00, a01], [a10, a11]], B11 = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)*(b00+b11)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[11][col] = 1
    W[0][col] = 1; W[2][col] = 1
    col += 1

    # M2 = (a10+a11)*b00
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[3][col] = 1
    col += 1

    # M3 = a00*(b01-b11)
    U[0][col] = 1
    V[1][col] = 1; V[11][col] = -1
    W[2][col] = 1
    col += 1

    # M4 = a11*(b10-b00)
    U[4][col] = 1
    V[10][col] = 1; V[0][col] = -1
    W[1][col] = 1
    col += 1

    # M5 = (a00+a01)*b11
    U[0][col] = 1; U[1][col] = 1
    V[11][col] = 1
    W[0][col] = -1; W[2][col] = 1
    col += 1

    # M6 = (a10-a00)*(b00+b01)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1
    col += 1

    # M7 = (a01-a11)*(b10+b11)
    U[1][col] = 1; U[4][col] = -1
    V[10][col] = 1; V[11][col] = 1
    W[0][col] = 1
    col += 1

    # Schoolbook for A2 (2x1) * B12 (1x2):
    # a02*b20
    U[2][col] = 1
    V[20][col] = 1
    W[0][col] = 1
    col += 1

    # a02*b21
    U[2][col] = 1
    V[21][col] = 1
    W[2][col] = 1
    col += 1

    # a12*b20
    U[5][col] = 1
    V[20][col] = 1
    W[1][col] = 1
    col += 1

    # a12*b21
    U[5][col] = 1
    V[21][col] = 1
    W[3][col] = 1
    col += 1

    # Part 2: Remaining 8 columns (k=2..9) using schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(2, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
