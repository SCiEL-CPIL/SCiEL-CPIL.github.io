import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 125  # safety margin before 150s limit

    n, m, p = DIMENSIONS
    # We will build a rank-16 decomposition:
    # For each j in 0..3 and each k_start in {0, 2, 4}:
    #   Use 1 rank-1 term for the 2x1x2 sub-tensor (rows i=0,1; col j; cols k_start, k_start+1)
    # For each j in 0..3 and k_start=6:
    #   Use 1 rank-1 term for the 2x1x1 sub-tensor
    # Total: 4 * (3 + 1) = 16

    rank = 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    for j in range(m):
        if time.time() > deadline:
            break
        # k_start in {0, 2, 4}: size-2 blocks
        for k_start in [0, 2, 4]:
            if time.time() > deadline:
                break
            # U: rows i=0,1 at column j -> indices i*m + j
            U[0 * m + j][col] += 1
            U[1 * m + j][col] += 1
            # V: row j at columns k_start, k_start+1 -> indices j*p + k
            V[j * p + k_start][col] += 1
            V[j * p + (k_start + 1)][col] += 1
            # W: columns k_start, k_start+1 at rows i=0,1 -> indices k*n + i
            W[k_start * n + 0][col] += 1
            W[k_start * n + 1][col] += 1
            W[(k_start + 1) * n + 0][col] += 1
            W[(k_start + 1) * n + 1][col] += 1
            col += 1
        # k_start=6: size-1 block
        if time.time() > deadline:
            break
        k = 6
        U[0 * m + j][col] += 1
        U[1 * m + j][col] += 1
        V[j * p + k][col] += 1
        W[k * n + 0][col] += 1
        W[k * n + 1][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
