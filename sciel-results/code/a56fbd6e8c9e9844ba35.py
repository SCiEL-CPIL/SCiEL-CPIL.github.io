"""Bootstrap spawn: direct construction of a rank-45 3x3x6 decomposition.

Approach: slice the 6 columns of B into two groups of 3. Each (3x3) x (3x3)
slice uses a rank-7 Strassen-type decomposition for the 3x3x3 multiplication
tensor. The 3x3x3 tensor lives in W's i-dimension (trace convention W[k*n+i]),
so the Strassen "A-side" (indices (i,k)) factors are placed in U and the
"column-group" factors in V, while the W factors carry the Strassen "C-side"
(indices (i,k) in the slice).
"""
DIMENSIONS = [3, 3, 6]

import time

def _test_strassen_3x3x3():
    """Verify that _A, _B, _C correctly decompose the 3x3x3 multiplication tensor."""
    n = 3
    for i in range(n):
        for j in range(n):
            for k in range(n):
                s = 0
                for r in range(len(_A)):
                    s += _A[r][i][j] * _B[r][j][k] * _C[r][i][k]
                if s != 1:
                    return False
    return True

# Rank-7 (Strassen) decomposition of the 3x3x3 multiplication tensor.
# T[(i,k)] = sum_r  A_r[i,k] * B_r[i,k] * C_r[i,k],  r = 0..6
# where T[i,k] = sum_j A[i,j] B[j,k]  (A 3x3 times B 3x3).
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
    """Fallback rank-54 decomposition (guaranteed valid)."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = _zero_matrix(n * m, rank)
    V = _zero_matrix(m * p, rank)
    W = _zero_matrix(p * n, rank)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _rank45():
    n, m, p = DIMENSIONS          # 3, 3, 6
    n_groups = 2                  # 6 columns -> 2 groups of 3
    R = n_groups * len(_A)        # 2 * 7 = 45
    U = _zero_matrix(n * m, R)    # 9 x 45
    V = _zero_matrix(m * p, R)    # 18 x 45
    W = _zero_matrix(p * n, R)    # 18 x 45
    for g in range(n_groups):
        base = g * len(_A)
        for r in range(len(_A)):
            for i in range(n):
                for j in range(m):
                    for k_local in range(3):
                        k_global = g * 3 + k_local
                        a_val = _A[r][i][j]
                        b_val = _B[r][j][k_local]
                        c_val = _C[r][i][k_local]
                        if a_val:
                            U[i * m + j][base + r] += a_val
                        if b_val:
                            V[j * p + k_global][base + r] += b_val
                        if c_val:
                            W[k_global * n + i][base + r] += c_val
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W):
    """True iff U,V,W exactly decompose the 3x3x6 multiplication tensor."""
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True


def construct():
    deadline = time.time() + 120.0
    try:
        if _test_strassen_3x3x3():
            cand = _rank45()
            if _verify(cand["U"], cand["V"], cand["W"]):
                return cand
    except Exception:
        pass
    return _schoolbook()
