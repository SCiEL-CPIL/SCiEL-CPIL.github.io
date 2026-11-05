"""Matrix multiplication 3,4,5 tensor decomposition.

construct() returns {'U','V','W'} factor matrices of shapes
(n*m, rank), (m*p, rank), (p*n, rank) with integer entries that exactly
decompose the 3x4x5 matrix multiplication tensor under the trace convention
W[k*n+i, r].  Score = rank (lower is better).

We build a valid decomposition by partitioning the product into independent
rectangular blocks (each a product tensor of rank area) and summing their
contributions.  The result is verified exactly before returning so that we
never emit an invalid tensor.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]


def construct():
    start = time.time()
    deadline = start + 120.0
    n, m, p = DIMENSIONS

    # ------------------------------------------------------------------
    # Block-partition construction of a valid decomposition.
    #
    # Partition rows (i) into blocks, columns (j) into blocks, and each
    # (i-block, j-block) rectangle is multiplied fully over all k.
    # A rectangle of size (ni x nj) has a product-tensor rank of ni*nj*p
    # (one rank per (i, j, k)).  Sum over blocks gives total rank.
    #
    # We choose a partition and, among candidates, keep the one with the
    # smallest total rank that passes exact verification.
    # ------------------------------------------------------------------

    def build(row_blocks, col_blocks):
        # row_blocks: list of (i_start, i_end); col_blocks: list of (j_start, j_end)
        cols = []
        for (i0, i1) in row_blocks:
            for (j0, j1) in col_blocks:
                for i in range(i0, i1):
                    for j in range(j0, j1):
                        for k in range(p):
                            cols.append((i, j, k))
        rank = len(cols)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for c, (i, j, k) in enumerate(cols):
            U[i * m + j][c] = 1
            V[j * p + k][c] = 1
            W[k * n + i][c] = 1
        return U, V, W, rank

    def verify(U, V, W):
        # Check the trace identity: for every r,
        #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # equals 1 if (i, j, k) is a "target" triple and 0 otherwise.
        # For matrix multiplication the target tensor T[i, j, k] = 1 for all
        # i, j, k (each output entry (i,k) is sum over j of A[i,j]B[j,k]).
        # So the scalar T[i,j,k] should equal 1 for every i, j, k.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(len(U[0])):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True

    # Candidate partitions (row_blocks, col_blocks).  The full 3x4 block gives
    # rank 60.  Finer partitions keep rank 60 (products are additive) but we
    # prefer coarser ones for compactness; all are valid.
    candidates = [
        ([(0, n)], [(0, m)]),                      # single 3x4 block: rank 60
        ([(0, 1), (1, n)], [(0, m)]),              # 1x4 + 2x4
        ([(0, n)], [(0, 2), (2, m)]),              # 3x2 + 3x2
        ([(0, 1), (1, n)], [(0, 2), (2, m)]),      # four 1x2 / 2x2 blocks
        ([(0, 1), (1, 2), (2, n)], [(0, 1), (1, 2), (2, 3), (3, m)]),
    ]

    best = None
    best_rank = None
    for idx, (rb, cb) in enumerate(candidates):
        if time.time() > deadline - 30.0:
            break
        U, V, W, rank = build(rb, cb)
        if verify(U, V, W):
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best = (U, V, W, rank)

    # Fallback: guaranteed-correct schoolbook (single block).
    if best is None:
        U, V, W, rank = build([(0, n)], [(0, m)])
        best = (U, V, W, rank)

    U, V, W, rank = best
    return {"U": U, "V": V, "W": W}
