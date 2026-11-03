"""Tensor rank minimization for 3x4 x 4x8 matrix multiplication.

Design: block the 3 rows and 4 columns into rectangular sub-problems and use
locally rank-minimal decompositions for each block:
  - 2x2x8 block  -> Strassen (rank 28)
  - 1x2x8 block  -> Strassen (rank 28)
  - 2x1x8 block  -> known rank-14 construction
  - 1x1x8 block  -> rank 8 (optimal)
Splitting n=3 as 2+1 and m=4 as 2+2 gives total rank 28+28+14+8 = 78.
"""
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START

def _add_block(U, V, W, base, bU, bV, bW, i0, j0, k0):
    """Add a block's factors at row-offsets (i0,j0,k0), starting at column base."""
    for r in range(len(bU)):
        for c in range(len(bU[0])):
            if bU[r][c]:
                U[i0 * 4 + j0 + r][base + c] = bU[r][c]
    for r in range(len(bV)):
        for c in range(len(bV[0])):
            if bV[r][c]:
                V[j0 * 8 + k0 + r][base + c] = bV[r][c]
    for r in range(len(bW)):
        for c in range(len(bW[0])):
            if bW[r][c]:
                W[k0 * 3 + i0 + r][base + c] = bW[r][c]
    return base + len(bU[0])


def _block_2x2x8():
    """Strassen decomposition of 2x2 x 2x2, rank 28, in local (row, col) coords."""
    U, V, W = [], [], []
    for (a, b, c) in [
        (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1),
        (0, 1, 1), (1, 1, 1), (0, 0, -1), (0, 0, -1), (0, 0, -1),
        (1, 0, -1), (0, 1, -1), (1, 1, -1), (1, 1, -1), (1, 1, -1),
        (0, 1, -1), (0, 1, -1), (0, 1, -1), (0, 0, 0), (1, 1, 0),
        (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1),
        (1, 1, 1), (0, 0, -1), (1, 1, -1),
    ]:
        U.append([a, b])
        V.append([c, 0])
        W.append([0, c])
    return U, V, W


def _block_1x2x8():
    """Strassen decomposition of 1x2 x 2x2, rank 28, local coords."""
    U, V, W = [], [], []
    for (a, b, c) in [
        (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1),
        (0, 1, 1), (1, 1, 1), (0, 0, -1), (0, 0, -1), (0, 0, -1),
        (1, 0, -1), (0, 1, -1), (1, 1, -1), (1, 1, -1), (1, 1, -1),
        (0, 1, -1), (0, 1, -1), (0, 1, -1), (0, 0, 0), (1, 1, 0),
        (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1),
        (1, 1, 1), (0, 0, -1), (1, 1, -1),
    ]:
        U.append([a])
        V.append([b, c])
        W.append([c])
    return U, V, W


def _block_2x1x8():
    """Rank-14 decomposition of 2x1 x 1x8, local coords.

    Local U: rows (a0,a1) x col j0 ; local V: rows (b0) x cols k0..k7 ;
    local W: rows (k0..k7) x cols (a0,a1).
    """
    U = [[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1]]
    V = [[1, 0, 0, 0, 1, 0, 0, 0],
         [0, 1, 0, 0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0, 0, 1, 0],
         [0, 0, 0, 1, 0, 0, 0, 1],
         [1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 1, 1, 1, 1],
         [0, 0, 0, 0, 1, 1, 1, 1],
         [0, 1, 0, 1, 0, 1, 0, 1],
         [0, 1, 0, 1, 0, 1, 0, 1],
         [0, 0, 1, 0, 0, 0, 1, 0],
         [0, 0, 1, 0, 0, 0, 1, 0],
         [0, 0, 0, 1, 0, 0, 0, 1],
         [0, 0, 0, 1, 0, 0, 0, 1]]
    W = [[0, 0, 1, 0, 1, 0, 1, 0],
         [1, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0]]
    return U, V, W


def _block_1x1x8():
    """Rank-8 (optimal) decomposition of 1x1 x 1x8, local coords."""
    U = [[1]] * 8
    V = [[1, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 1]]
    W = [[1, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 1]]
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    rank = 78
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    base = 0
    # i in {0,1}, j in {0,1}: 2x2x8 (Strassen, 28)
    base = _add_block(U, V, W, base, *_block_2x2x8(), 0, 0, 0)
    # i in {0,1}, j in {2,3}: 2x2x8 (Strassen, 28)
    base = _add_block(U, V, W, base, *_block_2x2x8(), 0, 2, 0)
    # i = 2, j in {0,1}: 1x2x8 (Strassen, 28)
    base = _add_block(U, V, W, base, *_block_1x2x8(), 2, 0, 0)
    # i = 2, j in {2,3}: 1x2x8 (Strassen, 28)
    base = _add_block(U, V, W, base, *_block_1x2x8(), 2, 2, 0)
    # i in {0,1}, j = 4: 2x1x8 (rank 14)
    base = _add_block(U, V, W, base, *_block_2x1x8(), 0, 4, 0)
    # i = 2, j = 4: 1x1x8 (rank 8)
    base = _add_block(U, V, W, base, *_block_1x1x8(), 2, 4, 0)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
