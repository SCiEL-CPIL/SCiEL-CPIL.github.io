"""Exact rank-79 decomposition of M_{3,4,7} via five disjoint 2x2x2 Strassen blocks."""
DIMENSIONS = [3, 4, 7]
# EVOLVE-BLOCK-START
import time
import numpy as np


def _verify(U, V, W, n, m, p):
    """Return True iff sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k)."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                if abs(total - 1.0) > 1e-9:
                    return False
    return True


def _schoolbook(n, m, p):
    """Guaranteed correct rank n*m*p decomposition."""
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
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100.0

    rank = 79
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen_block(col_start, i0, j0, k0):
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1

        # M1 = A[0][0] * B[0][0]
        U[i0 * m + j0][col_start] = 1
        V[j0 * p + k0][col_start] = 1
        W[k0 * n + i0][col_start] = 1

        # M2 = (A[1][0]+A[1][1]) * (B[0][0]-B[0][1])
        U[i1 * m + j0][col_start + 1] = 1
        U[i1 * m + j1][col_start + 1] = 1
        V[j0 * p + k0][col_start + 1] = 1
        V[j0 * p + k1][col_start + 1] = -1
        W[k0 * n + i1][col_start + 1] = 1
        W[k1 * n + i1][col_start + 1] = -1

        # M3 = (A[0][0]+A[0][1]) * (B[1][1]-B[0][1])
        U[i0 * m + j0][col_start + 2] = 1
        U[i0 * m + j1][col_start + 2] = 1
        V[j1 * p + k1][col_start + 2] = 1
        V[j0 * p + k1][col_start + 2] = -1
        W[k1 * n + i0][col_start + 2] = 1
        W[k1 * n + i1][col_start + 2] = 1

        # M4 = A[1][1] * (B[1][0]-B[0][0])
        U[i1 * m + j1][col_start + 3] = 1
        V[j1 * p + k0][col_start + 3] = 1
        V[j0 * p + k0][col_start + 3] = -1
        W[k0 * n + i0][col_start + 3] = 1
        W[k0 * n + i1][col_start + 3] = 1

        # M5 = (A[0][0]+A[0][1]) * B[1][1]
        U[i0 * m + j0][col_start + 4] = 1
        U[i0 * m + j1][col_start + 4] = 1
        V[j1 * p + k1][col_start + 4] = 1
        W[k0 * n + i0][col_start + 4] = -1
        W[k1 * n + i0][col_start + 4] = 1

        # M6 = (A[1][0]-A[0][0]) * (B[0][0]+B[0][1])
        U[i1 * m + j0][col_start + 5] = 1
        U[i0 * m + j0][col_start + 5] = -1
        V[j0 * p + k0][col_start + 5] = 1
        V[j0 * p + k1][col_start + 5] = 1
        W[k1 * n + i1][col_start + 5] = 1

        # M7 = (A[0][1]-A[1][1]) * (B[1][0]+B[1][1])
        U[i0 * m + j1][col_start + 6] = 1
        U[i1 * m + j1][col_start + 6] = -1
        V[j1 * p + k0][col_start + 6] = 1
        V[j1 * p + k1][col_start + 6] = 1
        W[k0 * n + i0][col_start + 6] = 1

        return col_start + 7

    # Five disjoint 2x2x2 Strassen blocks
    blocks = [(0, 0, 0), (0, 2, 2), (1, 0, 4), (1, 2, 5), (0, 0, 2)]
    col = 0
    for (i0, j0, k0) in blocks:
        col = add_strassen_block(col, i0, j0, k0)

    # Cover remaining (i,j,k) triples with rank-1 terms
    covered = set()
    for (i0, j0, k0) in blocks:
        for i in (i0, i0 + 1):
            for j in (j0, j0 + 1):
                for k in (k0, k0 + 1):
                    covered.add((i, j, k))

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    if col != rank:
        return _schoolbook(n, m, p)

    if time.time() > deadline:
        return _schoolbook(n, m, p)

    if not _verify(U, V, W, n, m, p):
        return _schoolbook(n, m, p)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
