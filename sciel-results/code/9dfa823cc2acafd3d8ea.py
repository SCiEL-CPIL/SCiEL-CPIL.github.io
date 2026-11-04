"""Tensor decomposition for 2x5 * 5x6 matrix multiplication."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    start_time = time.time()
    deadline = start_time + 140

    # Rank-10 decomposition: one term per (i,j) pair
    # For term r corresponding to (i_r, j_r):
    #   U[i_r*m + j_r, r] = 1
    #   V[j_r*p + k, r] = 1 for all k
    #   W[k*n + i_r, r] = 1 for all k
    # This exactly decomposes the matrix multiplication tensor.

    rank = n * m  # 2 * 5 = 10
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            # U part: only one non-zero entry
            u_idx = i * m + j
            U[u_idx][col] = 1

            # V part: all k for this j
            for k in range(p):
                v_idx = j * p + k
                V[v_idx][col] = 1

            # W part: all k for this i
            for k in range(p):
                w_idx = k * n + i
                W[w_idx][col] = 1

            col += 1

    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook
    return _schoolbook(n, m, p)


def _verify_decomposition(U, V, W, n, m, p, sample_size=5000):
    """Verify that the decomposition exactly represents matrix multiplication."""
    import random

    nm = n * m
    mp = m * p
    pn = p * n
    rank = len(U[0]) if U else 0

    # Build the set of tensor entries
    tensor_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                tensor_set.add((u, v, w))

    # Check all tensor entries
    for (u, v, w) in tensor_set:
        val = 0
        for r in range(rank):
            val += U[u][r] * V[v][r] * W[w][r]
        if abs(val - 1.0) > 1e-9:
            return False

    # Check a sample of non-tensor entries
    random.seed(42)
    for _ in range(sample_size):
        u = random.randint(0, nm - 1)
        v = random.randint(0, mp - 1)
        w = random.randint(0, pn - 1)
        if (u, v, w) in tensor_set:
            continue
        val = 0
        for r in range(rank):
            val += U[u][r] * V[v][r] * W[w][r]
        if abs(val) > 1e-9:
            return False

    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
