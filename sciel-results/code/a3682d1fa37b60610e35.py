"""Optimized rank-78 decomposition for 3x4*4x7 matrix multiplication.

Combines 6 disjoint 2x2x2 Strassen blocks with schoolbook for remaining entries.
This is the maximum number of disjoint 2x2x2 subcubes in a 3x4x7 grid (1*2*3=6).
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]
_DEADLINE = 140.0


def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS  # n=3, m=4, p=7

    rank = 6 * 7 + 36  # 78
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)

    col = 0

    def add_strassen_block(col_start, i0, i1, j0, j1, k0, k1):
        # M1: (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1])
        U[i0 * m + j0, col_start] = 1
        U[i1 * m + j1, col_start] = 1
        V[j0 * p + k0, col_start] = 1
        V[j1 * p + k1, col_start] = 1
        W[k0 * n + i0, col_start] = 1
        W[k1 * n + i1, col_start] = 1

        # M2: (A[i1,j0]+A[i1,j1])*B[j0,k0]
        U[i1 * m + j0, col_start + 1] = 1
        U[i1 * m + j1, col_start + 1] = 1
        V[j0 * p + k0, col_start + 1] = 1
        W[k0 * n + i1, col_start + 1] = 1
        W[k1 * n + i1, col_start + 1] = -1

        # M3: A[i0,j0]*(B[j0,k1]-B[j1,k1])
        U[i0 * m + j0, col_start + 2] = 1
        V[j0 * p + k1, col_start + 2] = 1
        V[j1 * p + k1, col_start + 2] = -1
        W[k1 * n + i0, col_start + 2] = 1
        W[k1 * n + i1, col_start + 2] = 1

        # M4: A[i1,j1]*(B[j1,k0]-B[j0,k0])
        U[i1 * m + j1, col_start + 3] = 1
        V[j1 * p + k0, col_start + 3] = 1
        V[j0 * p + k0, col_start + 3] = -1
        W[k0 * n + i0, col_start + 3] = 1
        W[k0 * n + i1, col_start + 3] = 1

        # M5: (A[i0,j0]+A[i0,j1])*B[j1,k1]
        U[i0 * m + j0, col_start + 4] = 1
        U[i0 * m + j1, col_start + 4] = 1
        V[j1 * p + k1, col_start + 4] = 1
        W[k0 * n + i0, col_start + 4] = -1
        W[k1 * n + i0, col_start + 4] = 1

        # M6: (A[i1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k1])
        U[i1 * m + j0, col_start + 5] = 1
        U[i0 * m + j0, col_start + 5] = -1
        V[j0 * p + k0, col_start + 5] = 1
        V[j0 * p + k1, col_start + 5] = 1
        W[k1 * n + i1, col_start + 5] = 1

        # M7: (A[i0,j1]-A[i1,j1])*(B[j1,k0]+B[j1,k1])
        U[i0 * m + j1, col_start + 6] = 1
        U[i1 * m + j1, col_start + 6] = -1
        V[j1 * p + k0, col_start + 6] = 1
        V[j1 * p + k1, col_start + 6] = 1
        W[k0 * n + i0, col_start + 6] = 1

        return col_start + 7

    blocks = [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 0, 1, 2, 3),
        (0, 1, 0, 1, 4, 5),
        (0, 1, 2, 3, 0, 1),
        (0, 1, 2, 3, 2, 3),
        (0, 1, 2, 3, 4, 5),
    ]

    covered = set()
    for (i0, i1, j0, j1, k0, k1) in blocks:
        col = add_strassen_block(col, i0, i1, j0, j1, k0, k1)
        for ii in (i0, i1):
            for jj in (j0, j1):
                for kk in (k0, k1):
                    covered.add((ii, jj, kk))

    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                if (i0, j0, k0) in covered:
                    continue
                U[i0 * m + j0, col] = 1
                V[j0 * p + k0, col] = 1
                W[k0 * n + i0, col] = 1
                col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"

    # Verify using vectorized numpy for speed
    if time.time() - _t0 < _DEADLINE - 5:
        # Tensor check: sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    total = np.sum(U[u_idx] * V[v_idx] * W[w_idx])
                    if total != 1:
                        raise ValueError(f"Decomposition failed at ({i},{j},{k}), got {total}")

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
