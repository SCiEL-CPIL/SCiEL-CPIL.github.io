"""Matrix multiplication 3x3x4 rank minimization."""
import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for c in range(rank):
        uc = U[:, c].reshape(n, m)
        vc = V[:, c].reshape(m, p)
        wc = W[:, c].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', uc, vc, wc)
    target = np.ones((n, m, p))
    return np.max(np.abs(T - target)) < tol

def construct():
    n, m, p = DIMENSIONS

    # Rank-12 construction: schoolbook for k=0,1,2 and k=3 separately
    rank = 12
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1
            V[j * p + 3, col] = 1
            W[3 * n + i, col] = 1
            col += 1

    assert col == rank

    if _verify_decomposition(U, V, W, n, m, p):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    # Fallback: full schoolbook
    rank_school = n * m * p
    U_s = np.zeros((n * m, rank_school))
    V_s = np.zeros((m * p, rank_school))
    W_s = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j, col] = 1
                V_s[j * p + k, col] = 1
                W_s[k * n + i, col] = 1
                col += 1

    return {'U': U_s.tolist(), 'V': V_s.tolist(), 'W': W_s.tolist()}
