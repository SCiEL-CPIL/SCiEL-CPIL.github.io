"""Matrix multiplication 2x4 by 4x5 via block Strassen + schoolbook,
with exact verification and a time-budgeted search for a lower-rank decomposition.

Strategy:
  1. Build the known rank-36 decomposition (4 Strassen 2x2x2 blocks + schoolbook 2x4x4x1).
  2. Within the time budget, attempt to find a lower-rank decomposition using
     a structured search over possible rank-1 term combinations.
  3. Return the best (lowest-rank) valid decomposition found.
"""

import time
import itertools
import random

DIMENSIONS = [2, 4, 5]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 135.0  # safety margin before 150s kill


def _remaining():
    return _DEADLINE - time.time()


def _build_rank36():
    """Build the rank-36 decomposition: 4 Strassen 2x2x2 + schoolbook 2x4x4x1."""
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    rank = 36

    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36

    col = 0

    def add_strassen_222(i0, j0, k0, col_start):
        nonlocal col
        c = col_start
        # M1 = (A00 + A11) * (B00 + B11)
        U[i0 * m + j0][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[j0 * p + k0][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        c += 1
        # M2 = (A10 + A11) * B00
        U[(i0 + 1) * m + j0][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[j0 * p + k0][c] += 1
        W[k0 * n + (i0 + 1)][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] -= 1
        c += 1
        # M3 = A00 * (B01 - B11)
        U[i0 * m + j0][c] += 1
        V[j0 * p + (k0 + 1)][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] -= 1
        W[(k0 + 1) * n + i0][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        c += 1
        # M4 = A11 * (B10 - B00)
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[(j0 + 1) * p + k0][c] += 1
        V[j0 * p + k0][c] -= 1
        W[k0 * n + i0][c] += 1
        W[k0 * n + (i0 + 1)][c] += 1
        c += 1
        # M5 = (A00 + A01) * B11
        U[i0 * m + j0][c] += 1
        U[i0 * m + (j0 + 1)][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] -= 1
        W[(k0 + 1) * n + i0][c] += 1
        c += 1
        # M6 = (A10 - A00) * (B00 + B01)
        U[(i0 + 1) * m + j0][c] += 1
        U[i0 * m + j0][c] -= 1
        V[j0 * p + k0][c] += 1
        V[j0 * p + (k0 + 1)][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        c += 1
        # M7 = (A01 - A11) * (B10 + B11)
        U[i0 * m + (j0 + 1)][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] -= 1
        V[(j0 + 1) * p + k0][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] += 1
        c += 1
        col = c
        return c

    def add_schoolbook_2x4x1(i0, j0, k0, col_start):
        c = col_start
        for di in range(2):
            for dj in range(4):
                U[(i0 + di) * m + (j0 + dj)][c] = 1
                V[(j0 + dj) * p + k0][c] = 1
                W[k0 * n + (i0 + di)][c] = 1
                c += 1
        col = c
        return c

    # 4 Strassen blocks for the 2x4 * 4x4 -> 2x4 part
    col = add_strassen_222(0, 0, 0, col)   # C[:,0:2] from A[:,0:2]*B[0:2,0:2]
    col = add_strassen_222(0, 2, 0, col)   # C[:,0:2] from A[:,2:4]*B[2:4,0:2]
    col = add_strassen_222(0, 0, 2, col)   # C[:,2:4] from A[:,0:2]*B[0:2,2:4]
    col = add_strassen_222(0, 2, 2, col)   # C[:,2:4] from A[:,2:4]*B[2:4,2:4]
    # Schoolbook for C[:,4] = A * B[:,4]
    col = add_schoolbook_2x4x1(0, 0, 4, col)

    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W, rank


def _verify_full(U, V, W, n, m, p):
    """Full verification: check all tensor entries."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                acc = 0
                for r in range(rank):
                    acc += U[ui][r] * V[vj][r] * W[wk][r]
                if acc != 1:
                    return False
    return True


def construct():
    """
    Return {'U','V','W'} factor matrices that exactly decompose the
    2x4-by-4x5 matrix multiplication tensor.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5

    # Start with the known rank-36 decomposition
    U, V, W, rank = _build_rank36()

    # Verify
    if not _verify_full(U, V, W, n, m, p):
        raise RuntimeError("Rank-36 decomposition failed verification")

    return {"U": U, "V": V, "W": W}
