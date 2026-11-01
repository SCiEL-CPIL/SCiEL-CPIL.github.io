import time
import numpy as np

DIMENSIONS = [2, 2, 12]

def _schoolbook(n=2, m=2, p=12):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _check(U, V, W, n, m, p, tol=1e-8):
    R = len(U[0]) if len(U)>0 else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - (1.0 if j == i else 0.0)) > tol:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_t = time.time()
    deadline = start_t + 105
    n, m, p = DIMENSIONS  # 2, 2, 12

    # Rank 2 decomposition
    rank = 2
    U = [[0.0] * rank for _ in range(n * m)]  # 4 x 2
    V = [[0.0] * rank for _ in range(m * p)]  # 24 x 2
    W = [[0.0] * rank for _ in range(p * n)]  # 24 x 2

    # U[i*m+j][r]
    # r=0 corresponds to j=0
    # r=1 corresponds to j=1
    for i in range(n):
        for j in range(m):
            U[i*m+j][j] = 1.0

    # V[j*p+k][r]
    # r=0: non-zero if j=0
    # r=1: non-zero if j=1
    for j in range(m):
        for k in range(p):
            V[j*p+k][j] = 1.0

    # W[k*n+i][r]
    # r=0: 1 for all k, i
    # r=1: 1 for all k, i
    for k in range(p):
        for i in range(n):
            W[k*n+i][0] = 1.0
            W[k*n+i][1] = 1.0

    # Verify
    if _check(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # Fallback
    fb = _schoolbook(n, m, p)
    return fb

# EVOLVE-BLOCK-END
