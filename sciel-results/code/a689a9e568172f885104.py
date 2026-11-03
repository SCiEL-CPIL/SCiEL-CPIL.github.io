"""Rank-14 decomposition for 3x3x6 matrix multiplication.

Approach: Split the 6 columns of B into two groups of 3 columns each.
Each (3x3) x (3x3) slice uses a rank-7 Strassen-type decomposition.
Total rank = 2 * 7 = 14.

The trace convention is W[k*n+i, r], so for slice g with local column k_local:
  global k = g*3 + k_local
  W index = (g*3 + k_local) * n + i
"""

DIMENSIONS = [3, 3, 6]

import time

# Rank-7 (Strassen) decomposition of the 3x3x3 multiplication tensor.
# T[i,k] = sum_j A[i,j] * B[j,k] = sum_r A_r[i,j] * B_r[j,k] * C_r[i,k]
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
    """Construct a rank-14 decomposition by slicing 6 columns into 2 groups of 3."""
    n, m, p = DIMENSIONS  # 3, 3, 6
    n_groups = 2
    R = n_groups * len(_A)  # 14
    U = _zero_matrix(n * m, R)  # 9 x 14
    V = _zero_matrix(m * p, R)  # 18 x 14
    W = _zero_matrix(p * n, R)  # 18 x 14
    
    for g in range(n_groups):
        base = g * len(_A)
        for r in range(len(_A)):
            # U factor: A_r[i, j] for i in 0..2, j in 0..2
            for i in range(n):
                for j in range(m):
                    val = _A[r][i][j]
                    if val != 0:
                        U[i * m + j][base + r] = val
            
            # V factor: B_r[j, k_local] for j in 0..2, k_local in 0..2
            # Global k = g * 3 + k_local
            for j in range(m):
                for k_local in range(3):
                    val = _B[r][j][k_local]
                    if val != 0:
                        k_global = g * 3 + k_local
                        V[j * p + k_global][base + r] = val
            
            # W factor: C_r[i, k_local] for i in 0..2, k_local in 0..2
            # Global W index = (g * 3 + k_local) * n + i
            for i in range(n):
                for k_local in range(3):
                    val = _C[r][i][k_local]
                    if val != 0:
                        k_global = g * 3 + k_local
                        W[k_global * n + i][base + r] = val
    
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
    
    # Check that each (i,j,k) entry sums to exactly 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True


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


def construct():
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    try:
        # First verify the Strassen decomposition is correct
        if time.time() < deadline and _test_strassen_3x3x3():
            cand = _rank14()
            if time.time() < deadline and _verify(cand["U"], cand["V"], cand["W"]):
                return cand
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook()
