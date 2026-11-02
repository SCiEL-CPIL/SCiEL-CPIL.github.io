"""Matrix multiplication tensor decomposition for 2x3 @ 3x8.

Block structure: 4 blocks of 2 k-columns each.
Each block handles A[:, k0:k0+2] @ B[k0:k0+2, :].

For the 2x2 sub-block (rows 0-1 of A, rows 0-1 of B), use Strassen (7 mults).
For the 2x1 sub-block (row 2 of A, row 2 of B), use 4 scalar mults.

Total: 4 blocks * 11 = 44 rank-1 terms.
"""

DIMENSIONS = [2, 3, 8]


def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    rank = 44  # 4 blocks * 11 multiplications per block

    # Initialize factor matrices as lists of lists of zeros
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    col = 0

    for b in range(4):
        # Two k-indices in this block
        k0 = 2 * b
        k1 = 2 * b + 1

        # ===== Strassen for A[0..1, 0..1] * B[0..1, k0..k1] (2x2 @ 2x2) =====
        # M1 = (A00 + A11) * (B0k0 + B1k1)
        U[0][col] = 1.0
        U[m + 1][col] = 1.0          # row 1, col 1 -> index 1*m+1 = 4
        V[0 * p + k0][col] = 1.0     # B[0, k0]
        V[1 * p + k1][col] = 1.0     # B[1, k1]
        W[k0 * n + 0][col] = 1.0     # C[0, k0] +=
        W[k1 * n + 1][col] = 1.0     # C[1, k1] +=
        col += 1

        # M2 = (A10 + A11) * B0k0
        U[m + 0][col] = 1.0          # A[1,0]
        U[m + 1][col] = 1.0          # A[1,1]
        V[0 * p + k0][col] = 1.0     # B[0, k0]
        W[k0 * n + 1][col] = 1.0     # C[1, k0] +=
        W[k1 * n + 1][col] = -1.0    # C[1, k1] -=
        col += 1

        # M3 = A00 * (B0k1 - B1k1)
        U[0][col] = 1.0              # A[0,0]
        V[0 * p + k1][col] = 1.0     # B[0, k1]
        V[1 * p + k1][col] = -1.0    # B[1, k1]
        W[k1 * n + 0][col] = 1.0     # C[0, k1] +=
        W[k1 * n + 1][col] = 1.0     # C[1, k1] +=
        col += 1

        # M4 = A11 * (B1k0 - B0k0)
        U[m + 1][col] = 1.0          # A[1,1]
        V[1 * p + k0][col] = 1.0     # B[1, k0]
        V[0 * p + k0][col] = -1.0    # B[0, k0]
        W[k0 * n + 0][col] = 1.0     # C[0, k0] +=
        W[k0 * n + 1][col] = 1.0     # C[1, k0] +=
        col += 1

        # M5 = (A00 + A01) * B1k1
        U[0][col] = 1.0              # A[0,0]
        U[1][col] = 1.0              # A[0,1]
        V[1 * p + k1][col] = 1.0     # B[1, k1]
        W[k0 * n + 0][col] = -1.0    # C[0, k0] -=
        W[k1 * n + 0][col] = 1.0     # C[0, k1] +=
        col += 1

        # M6 = (A10 - A00) * (B0k0 + B0k1)
        U[m + 0][col] = 1.0          # A[1,0]
        U[0][col] = -1.0             # A[0,0]
        V[0 * p + k0][col] = 1.0     # B[0, k0]
        V[0 * p + k1][col] = 1.0     # B[0, k1]
        W[k1 * n + 1][col] = 1.0     # C[1, k1] +=
        col += 1

        # M7 = (A01 - A11) * (B1k0 + B1k1)
        U[1][col] = 1.0              # A[0,1]
        U[m + 1][col] = -1.0         # A[1,1]
        V[1 * p + k0][col] = 1.0     # B[1, k0]
        V[1 * p + k1][col] = 1.0     # B[1, k1]
        W[k0 * n + 0][col] = 1.0     # C[0, k0] +=
        col += 1

        # ===== 4 scalar products for A[0..1, 2] * B[2, k0..k1] =====
        # P1 = A02 * B2k0
        U[2][col] = 1.0              # A[0,2]
        V[2 * p + k0][col] = 1.0     # B[2, k0]
        W[k0 * n + 0][col] = 1.0     # C[0, k0] +=
        col += 1

        # P2 = A02 * B2k1
        U[2][col] = 1.0              # A[0,2]
        V[2 * p + k1][col] = 1.0     # B[2, k1]
        W[k1 * n + 0][col] = 1.0     # C[0, k1] +=
        col += 1

        # P3 = A12 * B2k0
        U[m + 2][col] = 1.0          # A[1,2]
        V[2 * p + k0][col] = 1.0     # B[2, k0]
        W[k0 * n + 1][col] = 1.0     # C[1, k0] +=
        col += 1

        # P4 = A12 * B2k1
        U[m + 2][col] = 1.0          # A[1,2]
        V[2 * p + k1][col] = 1.0     # B[2, k1]
        W[k1 * n + 1][col] = 1.0     # C[1, k1] +=
        col += 1

    return {"U": U, "V": V, "W": W}
