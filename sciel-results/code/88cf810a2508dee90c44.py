"""Matrix multiplication 2x2 by 2x10 tensor decomposition (Strassen-based, rank 35)."""

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 10
    # We split the p=10 columns into 5 independent blocks of size 2.
    # For each block, we use Strassen's 7-term decomposition for 2x2 * 2x2.
    # Total rank = 5 * 7 = 35.

    rank = 35
    U = [[0] * rank for _ in range(n * m)]   # 4 x 35
    V = [[0] * rank for _ in range(m * p)]   # 20 x 35
    W = [[0] * rank for _ in range(p * n)]   # 20 x 35

    # Strassen's 7 multiplications for one 2x2 block.
    # A is 2x2 (rows i, cols j), B is 2x2 (rows j, cols k)
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B10
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # For each block b (0..4), columns k = 2*b, 2*b+1
    for b in range(5):
        base = b * 7
        k0 = 2 * b      # first column of this block
        k1 = 2 * b + 1  # second column of this block

        # s = 0..6 within this block
        # s=0: M1 = (A00+A11)*(B00+B11)
        # U: A00 -> U[0,0], A11 -> U[1,1]
        U[0][base + 0] = 1
        U[3][base + 0] = 1  # A11 is row 1, col 1 => index 1*2+1=3
        # V: B00 -> V[0,k0], B11 -> V[1,k1]
        V[0 * p + k0][base + 0] = 1
        V[1 * p + k1][base + 0] = 1
        # W: C00 gets M1, C11 gets M1
        # W[k*n + i, s]: k0 for C row 0, k1 for C row 1
        # C00: k=k0, i=0 => W[k0*n+0]
        # C11: k=k1, i=1 => W[k1*n+1]
        W[k0 * n + 0][base + 0] = 1
        W[k1 * n + 1][base + 0] = 1

        # s=1: M2 = (A10+A11)*B00
        U[2][base + 1] = 1  # A10: row 1, col 0 => index 1*2+0=2
        U[3][base + 1] = 1  # A11
        V[0 * p + k0][base + 1] = 1  # B00
        # C10 gets M2: k=k0, i=1 => W[k0*n+1]
        W[k0 * n + 1][base + 1] = 1

        # s=2: M3 = A00*(B01-B11)
        U[0][base + 2] = 1  # A00
        V[0 * p + k1][base + 2] = 1   # B01: row 0, col k1
        V[1 * p + k1][base + 2] = -1  # B11: row 1, col k1
        # C01 gets M3: k=k1, i=0 => W[k1*n+0]
        # C11 gets M3: k=k1, i=1 => W[k1*n+1]
        W[k1 * n + 0][base + 2] = 1
        W[k1 * n + 1][base + 2] = 1

        # s=3: M4 = A11*(B10-B00)
        U[3][base + 3] = 1  # A11
        V[1 * p + k0][base + 3] = 1   # B10: row 1, col k0
        V[0 * p + k0][base + 3] = -1  # B00: row 0, col k0
        # C00 gets M4: k=k0, i=0
        # C10 gets M4: k=k0, i=1
        W[k0 * n + 0][base + 3] = 1
        W[k0 * n + 1][base + 3] = 1

        # s=4: M5 = (A00+A01)*B10
        U[0][base + 4] = 1  # A00
        U[1][base + 4] = 1  # A01: row 0, col 1 => index 0*2+1=1
        V[1 * p + k0][base + 4] = 1  # B10
        # C00 gets -M5: k=k0, i=0
        # C01 gets M5: k=k1, i=0
        W[k0 * n + 0][base + 4] = -1
        W[k1 * n + 0][base + 4] = 1

        # s=5: M6 = (A10-A00)*(B00+B01)
        U[2][base + 5] = 1   # A10
        U[0][base + 5] = -1  # -A00
        V[0 * p + k0][base + 5] = 1  # B00
        V[0 * p + k1][base + 5] = 1  # B01
        # C11 gets M6: k=k1, i=1
        W[k1 * n + 1][base + 5] = 1

        # s=6: M7 = (A01-A11)*(B10+B11)
        U[1][base + 6] = 1   # A01
        U[3][base + 6] = -1  # -A11
        V[1 * p + k0][base + 6] = 1  # B10
        V[1 * p + k1][base + 6] = 1  # B11
        # C00 gets M7: k=k0, i=0
        W[k0 * n + 0][base + 6] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
