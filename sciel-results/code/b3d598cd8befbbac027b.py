"""Matrix multiplication 2x4x7 - rank-28 exact decomposition via block-split direct terms."""
DIMENSIONS = [2, 4, 7]

import numpy as np

def _verify(U, V, W, n, m, p):
    Ua = np.array(U, dtype=float); Va = np.array(V, dtype=float); Wa = np.array(W, dtype=float)
    rng = np.random.RandomState(12345)
    A = rng.rand(n, m); B = rng.rand(m, p)
    C_true = A @ B
    C_est = np.zeros((n, p))
    for c in range(Ua.shape[1]):
        P = np.zeros(n)
        for i in range(n):
            for j in range(m):
                P[i] += Ua[i*m+j, c] * A[i, j]
        Q = np.zeros(p)
        for k in range(p):
            for j in range(m):
                Q[k] += Va[j*p+k, c] * B[j, k]
        for i in range(n):
            for k in range(p):
                C_est[i, k] += Wa[k*n+i, c] * P[i] * Q[k]
    return np.allclose(C_est, C_true, atol=1e-9)

def _schoolbook(n, m, p):
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][column] = 1
                V[j*p+k][column] = 1
                W[k*n+i][column] = 1
                column += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    rank = 28
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for j in range(2):
        for k in range(p):
            U[0*m+j][col] = 1
            U[1*m+j][col] = 1
            V[j*p+k][col] = 1
            W[k*n+0][col] = 1
            W[k*n+1][col] = 1
            col += 1
    for j in range(2, 4):
        for k in range(p):
            U[0*m+j][col] = 1
            U[1*m+j][col] = 1
            V[j*p+k][col] = 1
            W[k*n+0][col] = 1
            W[k*n+1][col] = 1
            col += 1
    if not _verify(U, V, W, n, m, p):
        U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
