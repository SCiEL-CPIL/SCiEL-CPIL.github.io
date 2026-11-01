"""Matrix multiplication 2x3 by 3x8 via Strassen-like decomposition."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strategy: split B into 4 blocks of 3x2, apply rank-7 Strassen for 2x2*2x2
    # and rank-4 for 2x1*1x2 on each block. Total rank = 4*(7+4) = 44.
    rank = 44

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1

        # Strassen 7 products for A_top(2x2) * B_top(2x2)
        # A_top indices: (0,0),(0,1),(1,0),(1,1) -> flat: 0,1,3,4
        # B_top indices: (0,k0),(0,k1),(1,k0),(1,k1) -> flat: 0*p+k0, 0*p+k1, 1*p+k0, 1*p+k1

        # M1 = (A00+A11)*(B0k0+B1k1)
        U[0][col] = 1
        U[4][col] = 1
        V[0 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1

        # M2 = (A10+A11)*B0k0
        U[3][col] = 1
        U[4][col] = 1
        V[0 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1

        # M3 = A00*(B0k1-B1k1)
        U[0][col] = 1
        V[0 * p + k1][col] = 1
        V[1 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1

        # M4 = A11*(B1k0-B0k0)
        U[4][col] = 1
        V[1 * p + k0][col] = 1
        V[0 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1

        # M5 = (A00+A01)*B1k1
        U[0][col] = 1
        U[1][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1

        # M6 = (A10-A00)*(B0k0+B0k1)
        U[3][col] = 1
        U[0][col] = -1
        V[0 * p + k0][col] = 1
        V[0 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1

        # M7 = (A01-A11)*(B1k0+B1k1)
        U[1][col] = 1
        U[4][col] = -1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1

        # Rank-4 for A_bot(2x1) * B_bot(1x2)
        # A_bot: (0,2)=flat 2, (1,2)=flat 5
        # B_bot: (2,k0)=flat 2*p+k0, (2,k1)=flat 2*p+k1

        # P1 = A02*B2k0
        U[2][col] = 1
        V[2 * p + k0][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1

        # P2 = A02*B2k1
        U[2][col] = 1
        V[2 * p + k1][col] = 1
        W[k1 * n + 0][col] = 1
        col += 1

        # P3 = A12*B2k0
        U[5][col] = 1
        V[2 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1

        # P4 = A12*B2k1
        U[5][col] = 1
        V[2 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
