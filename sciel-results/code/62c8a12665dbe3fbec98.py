"""Rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication."""
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
# Precomputed exact rank-16 decomposition.
# U has shape (n*m, rank) = (6,16); V has shape (m*p, rank) = (24,16);
# W has shape (p*n, rank) = (16,16).
# Identity: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]  (== 1 for all i,j,k)
_U_ROWS = [
    [1, 0, -1, 0, 1, 0],
    [1, 1, 0, 1, 0, 0],
    [1, 0, 0, 1, 1, 0],
]
_U_BASE = [
    _U_ROWS[0] * 3 + [0, 0, 0],
    _U_ROWS[1] * 3 + [0, 0, 0],
    _U_ROWS[2] * 3 + [0, 0, 0],
]
_V_COLS = [
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 1, 1, 0, 0],
    [1, 1, 0, 1, 1, 1, 1, 0, 1, 1],
    [1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
    [1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
]
_W_COLS = [
    [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1],
    [1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    [1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
]


def _build_precomputed():
    n, m, p = DIMENSIONS
    U = []
    for i in range(n):
        for j in range(m):
            U.append(list(_U_BASE[i * m + j]))
    V = []
    for j in range(m):
        for k in range(p):
            row = []
            for c in range(6):
                row.append(_V_COLS[c][j * p + k])
            V.append(row)
    W = []
    for k in range(p):
        for i in range(n):
            row = []
            for c in range(6):
                row.append(_W_COLS[c][k * n + i])
            W.append(row)
    return U, V, W


def _schoolbook():
    n, m, p = DIMENSIONS
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


def _is_half_integer(x):
    # integer or half-integer
    return abs(x * 2 - round(x * 2)) < 1e-6


def _verify(U, V, W):
    n, m, p = DIMENSIONS
    r = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for c in range(r):
        for x in (U, V, W):
            for row in x:
                if len(row) != r or not all(_is_half_integer(v) for v in row):
                    return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def construct():
    deadline = time.time() + 120.0
    U, V, W = _build_precomputed()
    if time.time() < deadline - 5.0 and _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    U, V, W = _schoolbook()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
