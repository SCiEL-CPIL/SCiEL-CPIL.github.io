"""Rewritten program: ring-based decomposition for 2x2 x 2x12 matmul."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 12
    # Use Strassen-based decomposition: 6 pairs of columns, 7 multiplications each
    # Total rank = 42
    rank = 42
    U = [[0] * rank for _ in range(4)]   # n*m = 4
    V = [[0] * rank for _ in range(24)]  # m*p = 24
    W = [[0] * rank for _ in range(24)]  # p*n = 24

    col = 0
    for pair in range(6):
        k0 = 2 * pair
        k1 = 2 * pair + 1

        # S1 = (a+d) * (B[0,k0]+B[1,k0])
        # Contributes to C[0,k0] and C[1,k0]
        U[0][col] = 1
        U[3][col] = 1
        V[k0][col] = 1
        V[12 + k0][col] = 1
        W[2 * k0][col] = 1
        W[2 * k0 + 1][col] = 1
        col += 1

        # S2 = (c+d) * (B[0,k0]+B[0,k1])
        # Contributes to C[0,k0] and C[0,k1]
        U[2][col] = 1
        U[3][col] = 1
        V[k0][col] = 1
        V[k1][col] = 1
        W[2 * k0][col] = 1
        W[2 * k1][col] = 1
        col += 1

        # S3 = (a-b) * (B[1,k0]+B[1,k1])
        # Contributes to C[1,k0] and C[1,k1]
        U[0][col] = 1
        U[1][col] = -1
        V[12 + k0][col] = 1
        V[12 + k1][col] = 1
        W[2 * k0 + 1][col] = 1
        W[2 * k1 + 1][col] = 1
        col += 1

        # S4 = (a+c) * (B[0,k1]-B[1,k0])
        # Contributes to C[0,k1] only
        U[0][col] = 1
        U[2][col] = 1
        V[k1][col] = 1
        V[12 + k0][col] = -1
        W[2 * k1][col] = 1
        col += 1

        # S5 = (b+d) * (B[0,k0]-B[1,k1])
        # Contributes to C[0,k0] and C[1,k0]
        U[1][col] = 1
        U[3][col] = 1
        V[k0][col] = 1
        V[12 + k1][col] = -1
        W[2 * k0][col] = -1
        W[2 * k0 + 1][col] = 1
        col += 1

        # S6 = (a-c) * (B[0,k0]+B[1,k1])
        # Contributes to C[1,k1] only
        U[0][col] = 1
        U[2][col] = -1
        V[k0][col] = 1
        V[12 + k1][col] = 1
        W[2 * k1 + 1][col] = -1
        col += 1

        # S7 = (b-d) * (B[0,k1]+B[1,k1])
        # Contributes to C[0,k0] and C[1,k1]
        U[1][col] = 1
        U[3][col] = -1
        V[k1][col] = 1
        V[12 + k1][col] = 1
        W[2 * k0][col] = 1
        W[2 * k1 + 1][col] = -1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
