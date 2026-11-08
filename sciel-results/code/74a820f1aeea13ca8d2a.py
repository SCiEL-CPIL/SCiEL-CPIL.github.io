import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0

def _check_deadline():
    return (time.time() - _start_time) > (_DEADLINE - _SAFETY_MARGIN)

def _verify(U, V, W, n=2, m=3, p=5, tol=1e-9):
    """Verify exact decomposition."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    tensor = np.zeros((n, m, p))
    for r in range(rank):
        u = Ua[:, r].reshape(n, m)
        v = Va[:, r].reshape(m, p)
        w = Wa[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    return np.allclose(tensor, np.ones((n, m, p)), atol=tol)

def _naive(n=2, m=3, p=5):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    return {"U": U, "V": V, "W": W}

def construct():
    if _check_deadline():
        n, m, p = DIMENSIONS
        return _naive(n, m, p)
    
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    # Part 1: A2 B2 (j=2, k=0..4)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
    # Part 2: A1 B13 (j=0,1, k=4)
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
    # Part 3: Strassen for A1 B11
    U[0][r] = 1; U[4][r] = 1; V[0][r] = 1; V[6][r] = 1; W[0][r] = 1; W[5][r] = 1; r += 1
    U[3][r] = 1; U[4][r] = 1; V[0][r] = 1; W[1][r] = 1; W[5][r] = -1; r += 1
    U[0][r] = 1; V[1][r] = 1; V[6][r] = -1; W[4][r] = 1; W[5][r] = 1; r += 1
    U[4][r] = 1; V[5][r] = 1; V[0][r] = -1; W[0][r] = 1; W[1][r] = 1; r += 1
    U[0][r] = 1; U[1][r] = 1; V[6][r] = 1; W[0][r] = -1; W[4][r] = 1; r += 1
    U[3][r] = 1; U[0][r] = -1; V[0][r] = 1; V[1][r] = 1; W[5][r] = 1; r += 1
    U[1][r] = 1; U[4][r] = -1; V[5][r] = 1; V[6][r] = 1; W[0][r] = 1; r += 1
    # Part 4: Strassen for A1 B12
    U[0][r] = 1; U[4][r] = 1; V[2][r] = 1; V[8][r] = 1; W[2][r] = 1; W[7][r] = 1; r += 1
    U[3][r] = 1; U[4][r] = 1; V[2][r] = 1; W[3][r] = 1; W[7][r] = -1; r += 1
    U[0][r] = 1; V[3][r] = 1; V[8][r] = -1; W[6][r] = 1; W[7][r] = 1; r += 1
    U[4][r] = 1; V[7][r] = 1; V[2][r] = -1; W[2][r] = 1; W[3][r] = 1; r += 1
    U[0][r] = 1; U[1][r] = 1; V[8][r] = 1; W[2][r] = -1; W[6][r] = 1; r += 1
    U[3][r] = 1; U[0][r] = -1; V[2][r] = 1; V[3][r] = 1; W[7][r] = 1; r += 1
    U[1][r] = 1; U[4][r] = -1; V[7][r] = 1; V[8][r] = 1; W[2][r] = 1; r += 1
    
    if not _verify(U, V, W):
        return _naive(n, m, p)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
