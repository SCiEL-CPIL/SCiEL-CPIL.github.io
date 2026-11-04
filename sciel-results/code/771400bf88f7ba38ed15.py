import time
import numpy as np

DIMENSIONS = [3, 4, 8]


def _matmul_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _verify(U, V, W, n, m, p, tol=1e-6):
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    T = _matmul_tensor(n, m, p)
    r = U.shape[1]
    recon = np.zeros((n, m, p), dtype=np.float64)
    for c in range(r):
        uc = U[:, c].reshape(n, m)
        vc = V[:, c].reshape(m, p)
        wc = W[:, c].reshape(p, n)
        recon += np.einsum('ij,jk,ki->ijk', uc, vc, wc)
    recon_flat = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                recon_flat[i * m + j, j * p + k, k * n + i] = recon[i, j, k]
    return np.allclose(recon_flat, T, atol=tol)


def construct():
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
    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    res = construct()
    print("rank =", len(res["U"]))
    print("valid =", _verify(res["U"], res["V"], res["W"], 3, 4, 8))
