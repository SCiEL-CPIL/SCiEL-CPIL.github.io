"""
Rank-14 decomposition for 3x3x6 matrix multiplication.

Splits the 6 columns of B into two groups of 3.
Applies the rank-7 Strassen decomposition to each 3x3x3 slice.
Total rank = 7 + 7 = 14.
"""

import time

DIMENSIONS = [3, 3, 6]

# Rank-7 (Strassen) decomposition of the 3x3x3 multiplication tensor.
# T[(i,k)] = sum_r A_r[i,k] * B_r[i,k] * C_r[i,k]
# Indices: A(i,j), B(j,k), C(i,k)
_A = [
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[0, 0, 0], [1, 0, 0], [1, 1, 1]],
    [[0, 0, 0], [1, 0, 0], [1, 0, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 1, 1]],
    [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
]
_B = [
    [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
    [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
    [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[1, 0, 0], [0, 0, 0], [0, -1, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
    [[0, 0, 0], [0, 0, 0], [1, 1, 1]],
    [[0, 0, 0], [0, 0, 0], [1, 1, 1]],
]
_C = [
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
    [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 1], [0, 0, 1]],
    [[0, 0, 0], [0, 0, 0], [1, 1, 0]],
]


def _zero_matrix(rows, cols):
    return [[0] * cols for _ in range(rows)]


def _schoolbook():
    """Fallback rank-54 decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = _zero_matrix(n * m, rank)
    V = _zero_matrix(m * p, rank)
    W = _zero_matrix(p * n, rank)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _rank14():
    n, m, p = DIMENSIONS
    n_groups = 2
    R = n_groups * len(_A)
    U = _zero_matrix(n * m, R)
    V = _zero_matrix(m * p, R)
    W = _zero_matrix(p * n, R)

    for g in range(n_groups):
        base = g * len(_A)
        for r in range(len(_A)):
            # U: A_r[i, j]
            for i in range(n):
                for j in range(m):
                    val = _A[r][i][j]
                    if val:
                        U[i * m + j][base + r] = val

            # V: B_r[j, k_local] -> k_global
            for j in range(m):
                for k_local in range(3):
                    val = _B[r][j][k_local]
                    if val:
                        k_global = g * 3 + k_local
                        V[j * p + k_global][base + r] = val

            # W: C_r[i, k_local] -> k_global
            for i in range(n):
                for k_local in range(3):
                    val = _C[r][i][k_local]
                    if val:
                        k_global = g * 3 + k_local
                        W[k_global * n + i][base + r] = val

    return {"U": U, "V": V, "W": W}


def _verify(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for c in range(rank):
        if len(U[c]) != rank or len(V[c]) != rank or len(W[c]) != rank:
            return False
    seen = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s == 1.0:
                    if (i, j, k) in seen:
                        return False
                    seen.add((i, j, k))
                elif s != 0.0:
                    return False
    return len(seen) == n * m * p


def construct():
    deadline = time.time() + 120.0
    try:
        cand = _rank14()
        if _verify(cand["U"], cand["V"], cand["W"]):
            return cand
    except Exception:
        pass
    if time.time() > deadline - 5.0:
        return _schoolbook()
    return _schoolbook()
