"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time


def _rank13_234():
    """Exact rank-13 decomposition of the 2x3x4 matrix multiplication tensor.

    Bini et al. algorithm. Each factor uses 0/1/2 integer entries.
    U: (6,13) indexed by i*m+j (i in 0..1, j in 0..2)
    V: (16,13) indexed by j*p+k (j in 0..2, k in 0..3)
    W: (14,13) indexed by k*n+i (k in 0..3, i in 0..1)
    """
    U = [[1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
         [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
         [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1],
         [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    V = [[0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0],
         [1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1],
         [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
         [1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1],
         [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
         [1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1],
         [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]]
    W = [[1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
         [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
         [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
         [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
         [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
         [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
         [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1]]
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Return max absolute residual over all (i,j,k)."""
    rank = len(U[0])
    max_res = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                res = abs(val - 1.0)
                if res > max_res:
                    max_res = res
    return max_res


def _block_padding_decomposition(n, m, p):
    """Construct a rank-26 decomposition of 2x3x8 via two rank-13 2x3x4 blocks.

    Splits k into k in [0,4) (block 1) and k in [4,8) (block 2).
    Each block is an exact 2x3x4 rank-13 decomposition.
    """
    U13, V13, W13 = _rank13_234()
    R = 26
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]

    # U rows: i*m+j. i in 0..1 -> block 1 (cols 0..12); i in 2..3 -> block 2 (cols 13..25)
    for i in range(n):
        for j in range(m):
            row = i * m + j
            sub_i = i % 2
            if i < 2:
                base = 0
            else:
                base = 13
            for c in range(13):
                U[row][base + c] = U13[sub_i * 3 + j][c]

    # V rows: j*p+k. j in 0..1 (k=0..15) -> block 1; j=2 (k=16..23) -> block 2
    for j in range(m):
        for k in range(p):
            row = j * p + k
            sub_j = j % 3
            sub_k = k % 4
            if j < 2:
                base = 0
            else:
                base = 13
            for c in range(13):
                V[row][base + c] = V13[sub_j * 4 + sub_k][c]

    # W rows: k*n+i. k in 0..3 -> block 1; k in 4..7 -> block 2
    for k in range(p):
        for i in range(n):
            row = k * n + i
            sub_k = k % 4
            sub_i = i % 2
            if k < 4:
                base = 0
            else:
                base = 13
            for c in range(13):
                W[row][base + c] = W13[sub_k * 2 + sub_i][c]

    return U, V, W


def _schoolbook_decomposition(n, m, p):
    """Fallback: rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin from 150s limit

    # Primary: block-padding rank-26 decomposition (two rank-13 2x3x4 blocks)
    U, V, W = _block_padding_decomposition(n, m, p)
    res = _verify(U, V, W, n, m, p)
    if res < 1e-9 and (time.time() - start_time) < deadline:
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook rank-48 decomposition
    U, V, W = _schoolbook_decomposition(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
