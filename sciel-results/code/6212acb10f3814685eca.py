import numpy as np
import time

_DIMS = (4, 4, 7)
_START = time.time()

def _deadline(sec):
    return time.time() - _START < sec

def _mat2rows(M):
    n, m = M.shape
    return [M[i, j] for i in range(n) for j in range(m)]

def _verify(U, V, W, n, m, p):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    T = np.einsum('ar,br,cr->abc', U, V, W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T[i * m + j, j * p + k, k * n + i] != 1.0:
                    return False
    if not np.allclose(T, np.round(T), atol=1e-9):
        return False
    nz = T[np.abs(T) > 1e-9]
    if not np.allclose(nz, np.round(nz), atol=1e-9):
        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1
    return {"U": U, "V": V, "W": W}

def _strassen222():
    U = [[1,0,1,0],[0,1,0,0],[0,0,1,1],[1,0,0,1]]
    V = [[1,1,0,0],[0,0,1,1],[1,0,1,0],[0,1,0,-1]]
    W = [[1,0,0,1],[1,1,0,0],[0,0,1,0],[0,1,1,-1]]
    return U, V, W

def _factor227():
    # 2x2x7: B has 7 cols. cols 0,1 Strassen (rank7); cols 2..6 schoolbook (rank5) => rank 12.
    U2, V2, W2 = _strassen222()
    U = []
    V = []
    W = []
    # Strassen block on cols 0,1 (B index = j*7 + k)
    for r in range(7):
        u = list(U2[:, r])
        v = [0.0, 0.0, 0.0, 0.0]
        v[0 * 7 + 0] = V2[0, r]
        v[0 * 7 + 1] = V2[1, r]
        v[1 * 7 + 0] = V2[2, r]
        v[1 * 7 + 1] = V2[3, r]
        w = list(W2[:, r])
        U.append(u); V.append(v); W.append(w)
    # Schoolbook on cols 2..6
    for j in range(2):
        for k in range(2, 7):
            u = [0.0, 0.0, 0.0, 0.0]
            v = [0.0, 0.0, 0.0, 0.0]
            w = [0.0, 0.0, 0.0, 0.0]
            u[j] = 1.0
            v[j * 7 + k] = 1.0
            w[j] = 1.0
            U.append(u); V.append(v); W.append(w)
    return np.array(U).T, np.array(V).T, np.array(W).T

def _factor221():
    # 2x2x1: rank 2 schoolbook
    U = [[1.0, 0.0], [0.0, 1.0]]
    V = [[1.0, 1.0]]
    W = [[1.0, 1.0]]
    return U, V, W

def construct():
    n, m, p = _DIMS
    if not _deadline(140):
        return _schoolbook(n, m, p)

    # Row-block factor (2x2x7) and col-block factor (2x2x1)
    Fu, Fv, Fw = _factor227()
    Gu, Gv, Gw = _factor221()
    Rf = Fu.shape[1]
    Rg = Gu.shape[1]
    R = Rf * Rg

    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]

    for a in range(Rf):
        for b in range(Rg):
            r = a * Rg + b
            u = _mat2rows(np.kron(Gu[:, b], Fu[:, a]))
            v = _mat2rows(np.kron(Gv[:, b], Fv[:, a]))
            w = _mat2rows(np.kron(Gw[:, b], Fw[:, a]))
            for idx in range(n * m):
                U[idx][r] = u[idx]
            for idx in range(m * p):
                V[idx][r] = v[idx]
            for idx in range(p * n):
                W[idx][r] = w[idx]

    if not _verify(U, V, W, n, m, p):
        return _schoolbook(n, m, p)

    return {"U": U, "V": V, "W": W}
