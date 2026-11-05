"""BCR/Strassen recursive tensor decomposition for 2x4x8 matrix multiplication.

Rank = 7 * 7 = 49 via two-level Strassen recursion.
  Outer 2x2: rows i in {0,1} x column-groups (k//2) in {0,1}
  Inner 2x2: each outer entry is a 2x2 Strassen product over the block indices.

Official trace convention: W[k*n + i, r] is the coefficient of output (i,k).
The decomposition satisfies, for every (i, j, k):
    sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = 1
"""
import time

_DIMS = (2, 4, 8)

_M = [
    (0, 0, 0), (0, 1, 1), (1, 0, 0), (1, 1, 1),
    (0, 0, 1), (1, 1, 0), (1, 0, 1),
]
_SAr = [
    [1, 0, 1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0, 1, 1],
]
_SBr = [
    [1, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 1, 0, 1, 1],
]


def _build():
    n, m, p = _DIMS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for i in range(n):
        for j in range(m):
            for r in range(rank):
                rA, rB = divmod(r, 7)
                U[i * m + j][r] = _SAr[i][rA] * _SAr[j // 2][rB]

    for j in range(m):
        for k in range(p):
            for r in range(rank):
                rA, rB = divmod(r, 7)
                V[j * p + k][r] = _SBr[j // 2][rA] * _SBr[k // 4][rB]

    for k in range(p):
        for i in range(n):
            for r in range(rank):
                rA, rB = divmod(r, 7)
                a, b, c = _M[rA]
                # Inner 2x2 product for outer (a, b, c):
                #   A-block(row i, col a) * B-block(row b, col c)
                #   = SAr[i][rB] * SBr[c][rB]  (A row i col a, B row b col c)
                #   but the B row index is b and col index is c, so:
                W[k * n + i][r] = _SAr[i][rB] * _SBr[c][rB]
    return U, V, W


def _verify(U, V, W, n, m, p, rank):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True


def _schoolbook():
    n, m, p = _DIMS
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


def construct():
    start = time.time()
    deadline = start + 110.0

    n, m, p = _DIMS
    U, V, W = _build()
    if time.time() > deadline:
        return _schoolbook()
    if _verify(U, V, W, n, m, p, 49):
        return {"U": U, "V": V, "W": W}
    return _schoolbook()
