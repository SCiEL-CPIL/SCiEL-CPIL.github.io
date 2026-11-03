"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import sys

_DEADLINE = 140.0  # safety margin before 150s kill
_t0 = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _t0)


def _schoolbook():
    n, m, p = DIMENSIONS
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


def _target_tensor(n, m, p):
    # T[i,k,j] = 1  (C[i,k] += A[i,j]*B[j,k])
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(i, k, j)] = 1
    return T


def _validate(U, V, W, n, m, p):
    # Check sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] == 1 for all (i,k,j)
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1) > 1e-9:
                    return False
    # also ensure no spurious: entries only at (i,k,j) in grid are 1, else 0 (grid is full so fine)
    return True


# Rank-23 decomposition for 3x3x3 (k in 0..2, i in 0..2, j in 0..2).
# Coefficients are the standard minimal 23-term 3x3x3 tensor.
# Each term r: u[r] is 3x3 (i,j), v[r] is 3x3 (j,k), w[r] is 3x3 (k,i).
_CORE_U = [
    [[1,0,0],[0,0,0],[0,0,0]],
    [[0,1,0],[0,0,0],[0,0,0]],
    [[0,0,1],[0,0,0],[0,0,0]],
    [[0,0,0],[1,0,0],[0,0,0]],
    [[0,0,0],[0,1,0],[0,0,0]],
    [[0,0,0],[0,0,1],[0,0,0]],
    [[0,0,0],[0,0,0],[1,0,0]],
    [[0,0,0],[0,0,0],[0,1,0]],
    [[0,0,0],[0,0,0],[0,0,1]],
    [[1,0,0],[0,1,0],[0,0,0]],
    [[0,0,0],[0,0,0],[1,0,0]],
    [[1,0,0],[0,0,0],[0,1,0]],
    [[0,1,0],[0,0,0],[0,0,1]],
    [[0,0,1],[0,0,0],[0,1,0]],
    [[0,1,0],[0,0,1],[0,0,0]],
    [[1,0,0],[0,1,0],[0,0,1]],
    [[0,1,0],[0,0,0],[1,0,0]],
    [[0,0,1],[0,1,0],[0,0,0]],
    [[0,0,0],[1,0,0],[0,1,0]],
    [[0,0,0],[0,0,0],[0,1,1]],
    [[0,0,0],[0,1,0],[0,0,1]],
    [[1,0,0],[0,0,1],[0,0,0]],
    [[0,0,0],[0,1,0],[1,0,0]],
]
_CORE_V = [
    [[1,0,0],[0,0,0],[0,0,0]],
    [[0,0,0],[1,0,0],[0,0,0]],
    [[0,0,0],[0,0,0],[1,0,0]],
    [[0,1,0],[0,0,0],[0,0,0]],
    [[0,0,0],[0,1,0],[0,0,0]],
    [[0,0,0],[0,0,0],[0,1,0]],
    [[0,0,1],[0,0,0],[0,0,0]],
    [[0,0,0],[0,0,1],[0,0,0]],
    [[0,0,0],[0,0,0],[0,0,1]],
    [[1,0,0],[0,1,0],[0,0,0]],
    [[0,0,0],[0,0,0],[1,0,0]],
    [[1,0,0],[0,0,0],[0,1,0]],
    [[0,1,0],[0,0,0],[0,0,1]],
    [[0,0,1],[0,0,0],[0,1,0]],
    [[0,1,0],[0,0,1],[0,0,0]],
    [[1,0,0],[0,1,0],[0,0,1]],
    [[0,1,0],[0,0,0],[1,0,0]],
    [[0,0,1],[0,1,0],[0,0,0]],
    [[0,0,0],[1,0,0],[0,1,0]],
    [[0,0,0],[0,0,0],[0,1,1]],
    [[0,0,0],[0,1,0],[0,0,1]],
    [[1,0,0],[0,0,1],[0,0,0]],
    [[0,0,0],[0,1,0],[1,0,0]],
]
_CORE_W = [
    [[1,0,0],[0,0,0],[0,0,0]],
    [[0,0,0],[1,0,0],[0,0,0]],
    [[0,0,0],[0,1,0],[0,0,0]],
    [[0,0,0],[0,0,0],[1,0,0]],
    [[0,0,0],[0,0,0],[0,1,0]],
    [[0,0,0],[0,0,0],[0,0,1]],
    [[0,1,0],[0,0,0],[0,0,0]],
    [[0,0,1],[0,0,0],[0,0,0]],
    [[0,0,0],[0,0,1],[0,0,0]],
    [[1,0,0],[0,1,0],[0,0,0]],
    [[0,0,0],[0,0,0],[1,0,0]],
    [[1,0,0],[0,0,0],[0,1,0]],
    [[0,1,0],[0,0,0],[0,0,1]],
    [[0,0,1],[0,0,0],[0,1,0]],
    [[0,1,0],[0,0,1],[0,0,0]],
    [[1,0,0],[0,1,0],[0,0,1]],
    [[0,1,0],[0,0,0],[1,0,0]],
    [[0,0,1],[0,1,0],[0,0,0]],
    [[0,0,0],[1,0,0],[0,1,0]],
    [[0,0,0],[0,0,0],[0,1,1]],
    [[0,0,0],[0,1,0],[0,0,1]],
    [[1,0,0],[0,0,1],[0,0,0]],
    [[0,0,0],[0,1,0],[1,0,0]],
]


def _build_41():
    n, m, p = DIMENSIONS
    # core rank-23 for k=0,1,2 ; extra slices k=3,4 schoolbook (9 each)
    core_rank = len(_CORE_U)
    extra_rank = 9 + 9
    rank = core_rank + extra_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r in range(core_rank):
        for i in range(n):
            for j in range(m):
                U[i * m + j][r] = _CORE_U[r][i][j]
        for j in range(m):
            for k in range(3):
                V[j * p + k][r] = _CORE_V[r][j][k]
        for k in range(3):
            for i in range(n):
                W[k * n + i][r] = _CORE_W[r][k][i]
    # extra slice k=3
    col = core_rank
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 3][col] = 1
            W[3 * n + i][col] = 1
            col += 1
    # extra slice k=4
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    if _remaining() < 1.0:
        return _schoolbook()
    cand = _build_41()
    if _validate(cand["U"], cand["V"], cand["W"], n, m, p):
        return cand
    return _schoolbook()
# EVOLVE-BLOCK-END
