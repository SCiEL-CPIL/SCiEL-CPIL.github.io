"""Rewritten program for 2x3x8 matrix multiplication tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 8]


def construct():
    n, m, p = DIMENSIONS

    # We'll build a rank-44 decomposition using a block strategy:
    # Split p=8 into two groups of 4 columns.
    # Within each 3x4 block, split m=3 into m=2 and m=1.
    # The m=2 x p=4 part uses Strassen (rank 7+7=14).
    # The m=1 x p=4 part uses schoolbook (rank 8).
    # Total: 2*(14+8) = 44.

    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def emit_strassen_block(i0, j0, k0):
        """Emit a 2x2x2 Strassen decomposition at offsets (i0,j0,k0)."""
        nonlocal col
        # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
        # U row index: (i0+i)*m + (j0+j)
        # V row index: (j0+j)*p + (k0+k)
        # W row index: (k0+k)*n + (i0+i)

        # M1 = (a11+a22)(b11+b22)
        U[(i0 + 0) * m + (j0 + 0)][col] = 1
        U[(i0 + 1) * m + (j0 + 1)][col] = 1
        V[(j0 + 0) * p + (k0 + 0)][col] = 1
        V[(j0 + 1) * p + (k0 + 1)][col] = 1
        W[(k0 + 0) * n + (i0 + 0)][col] = 1
        W[(k0 + 1) * n + (i0 + 1)][col] = 1
        col += 1

        # M2 = (a21+a22) b11
        U[(i0 + 1) * m + (j0 + 0)][col] = 1
        U[(i0 + 1) * m + (j0 + 1)][col] = 1
        V[(j0 + 0) * p + (k0 + 0)][col] = 1
        W[(k0 + 0) * n + (i0 + 1)][col] = 1
        W[(k0 + 1) * n + (i0 + 1)][col] = -1
        col += 1

        # M3 = a11 (b21 - b11)
        U[(i0 + 0) * m + (j0 + 0)][col] = 1
        V[(j0 + 0) * p + (k0 + 1)][col] = -1
        V[(j0 + 1) * p + (k0 + 1)][col] = 1
        W[(k0 + 1) * n + (i0 + 0)][col] = 1
        W[(k0 + 1) * n + (i0 + 1)][col] = 1
        col += 1

        # M4 = a22 (b12 - b22)
        U[(i0 + 1) * m + (j0 + 1)][col] = 1
        V[(j0 + 0) * p + (k0 + 0)][col] = -1
        V[(j0 + 1) * p + (k0 + 0)][col] = 1
        W[(k0 + 0) * n + (i0 + 0)][col] = 1
        W[(k0 + 0) * n + (i0 + 1)][col] = 1
        col += 1

        # M5 = (a11+a12) b22
        U[(i0 + 0) * m + (j0 + 0)][col] = 1
        U[(i0 + 0) * m + (j0 + 1)][col] = 1
        V[(j0 + 1) * p + (k0 + 1)][col] = 1
        W[(k0 + 0) * n + (i0 + 0)][col] = -1
        W[(k0 + 1) * n + (i0 + 0)][col] = 1
        col += 1

        # M6 = (a21 - a11)(b11 + b12)
        U[(i0 + 1) * m + (j0 + 0)][col] = 1
        U[(i0 + 0) * m + (j0 + 0)][col] = -1
        V[(j0 + 0) * p + (k0 + 0)][col] = 1
        V[(j0 + 0) * p + (k0 + 1)][col] = 1
        W[(k0 + 1) * n + (i0 + 1)][col] = 1
        col += 1

        # M7 = (a12 - a22)(b21 + b22)
        U[(i0 + 0) * m + (j0 + 1)][col] = 1
        U[(i0 + 1) * m + (j0 + 1)][col] = -1
        V[(j0 + 1) * p + (k0 + 0)][col] = 1
        V[(j0 + 1) * p + (k0 + 1)][col] = 1
        W[(k0 + 0) * n + (i0 + 0)][col] = 1
        col += 1

    def emit_schoolbook_slice(i0, j0, k0, num_k):
        """Emit schoolbook for n x 1 x num_k slice at offsets (i0, j0, k0)."""
        nonlocal col
        for i in range(n):
            for k in range(num_k):
                U[(i0 + i) * m + (j0 + 0)][col] = 1
                V[(j0 + 0) * p + (k0 + k)][col] = 1
                W[(k0 + k) * n + (i0 + i)][col] = 1
                col += 1

    # First 3x4 block: k in [0, 4)
    # m=2 part: j in [0,2), k in [0,2) -> Strassen
    emit_strassen_block(0, 0, 0)
    # m=2 part: j in [0,2), k in [2,4) -> Strassen
    emit_strassen_block(0, 0, 2)
    # m=1 part: j=2, k in [0,4) -> schoolbook
    emit_schoolbook_slice(0, 2, 0, 4)

    # Second 3x4 block: k in [4, 8)
    # m=2 part: j in [0,2), k in [4,6) -> Strassen
    emit_strassen_block(0, 0, 4)
    # m=2 part: j in [0,2), k in [6,8) -> Strassen
    emit_strassen_block(0, 0, 6)
    # m=1 part: j=2, k in [4,8) -> schoolbook
    emit_schoolbook_slice(0, 2, 4, 4)

    return {"U": U, "V": V, "W": W}
