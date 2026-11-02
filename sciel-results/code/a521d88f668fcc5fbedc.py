"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 44

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # We split the 3x8 B matrix into four 3x2 blocks.
    # For each 3x2 block, we use an 11-rank decomposition for 2x3 * 3x2.
    # This gives 4 * 11 = 44 total rank.

    for b in range(4):
        k0 = 2 * b
        k1 = k0 + 1

        # --- Strassen part for A1 (2x2) * B1 (2x2) ---
        # A1 = [[A00, A01], [A10, A11]] = [[a, b], [c, d]]
        # B1 = [[B0k0, B0k1], [B1k0, B1k1]] = [[e, f], [g, h]]
        # C1 = A1 * B1

        # M1 = (a + d) * (e + h)
        U[0 * m + 0][col] = 1  # A00
        U[1 * m + 1][col] = 1  # A11
        V[0 * p + k0][col] = 1 # B0k0
        V[1 * p + k1][col] = 1 # B1k1
        # W contributions: C0k0 += 1, C1k1 += 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1

        # M2 = (c + d) * e
        U[1 * m + 0][col] = 1  # A10
        U[1 * m + 1][col] = 1  # A11
        V[0 * p + k0][col] = 1 # B0k0
        # W contributions: C1k0 += 1, C1k1 += -1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1

        # M3 = a * (f - h)
        U[0 * m + 0][col] = 1  # A00
        V[0 * p + k1][col] = 1 # B0k1
        V[1 * p + k1][col] = -1 # B1k1
        # W contributions: C0k1 += 1, C1k1 += 1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1

        # M4 = d * (g - e)
        U[1 * m + 1][col] = 1  # A11
        V[1 * p + k0][col] = 1 # B1k0
        V[0 * p + k0][col] = -1 # B0k0
        # W contributions: C0k0 += 1, C1k0 += 1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1

        # M5 = (a + b) * g
        U[0 * m + 0][col] = 1  # A00
        U[0 * m + 1][col] = 1  # A01
        V[1 * p + k0][col] = 1 # B1k0
        # W contributions: C0k0 += -1, C0k1 += 1
        W[k0 * n + 0][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1

        # M6 = (c - a) * (e + f)
        U[1 * m + 0][col] = 1  # A10
        U[0 * m + 0][col] = -1 # A00
        V[0 * p + k0][col] = 1 # B0k0
        V[0 * p + k1][col] = 1 # B0k1
        # W contributions: C1k1 += 1
        W[k1 * n + 1][col] = 1
        col += 1

        # M7 = (b - d) * (g + h)
        U[0 * m + 1][col] = 1  # A01
        U[1 * m + 1][col] = -1 # A11
        V[1 * p + k0][col] = 1 # B1k0
        V[1 * p + k1][col] = 1 # B1k1
        # W contributions: C0k0 += 1
        W[k0 * n + 0][col] = 1
        col += 1

        # --- Rank-4 part for A2 (2x1) * B2 (1x2) ---
        # A2 = [A02, A12] = [c2, d2]
        # B2 = [B2k0, B2k1] = [e2, f2]
        # C2 = A2 * B2 (outer product)

        # P1 = A02 * B2k0
        U[0 * m + 2][col] = 1  # A02
        V[2 * p + k0][col] = 1 # B2k0
        # W contributions: C0k0 += 1
        W[k0 * n + 0][col] = 1
        col += 1

        # P2 = A02 * B2k1
        U[0 * m + 2][col] = 1  # A02
        V[2 * p + k1][col] = 1 # B2k1
        # W contributions: C0k1 += 1
        W[k1 * n + 0][col] = 1
        col += 1

        # P3 = A12 * B2k0
        U[1 * m + 2][col] = 1  # A12
        V[2 * p + k0][col] = 1 # B2k0
        # W contributions: C1k0 += 1
        W[k0 * n + 1][col] = 1
        col += 1

        # P4 = A12 * B2k1
        U[1 * m + 2][col] = 1  # A12
        V[2 * p + k1][col] = 1 # B2k1
        # W contributions: C1k1 += 1
        W[k1 * n + 1][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
