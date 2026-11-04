import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]
_DEADLINE = 140.0

def _remaining():
    return _DEADLINE - time.time()

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, U, V, W):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _verify_exact(n, m, p, U, V, W):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) * Fraction(W[k * n + i][c])
                if s != 1:
                    return False
    return True

def _als(n, m, p, rank, seed, max_iter=200):
    np.random.seed(seed)
    U = np.random.randn(n * m, rank) * 0.3
    V = np.random.randn(m * p, rank) * 0.3
    W = np.random.randn(p * n, rank) * 0.3

    for it in range(max_iter):
        if time.time() > _DEADLINE - 10:
            break

        for i in range(n):
            for j in range(m):
                ui = i * m + j
                A = np.array([V[j * p + k] * W[k * n + i] for k in range(p)])
                try:
                    U[ui] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                except:
                    pass

        for j in range(m):
            for k in range(p):
                vi = j * p + k
                A = np.array([U[i * m + j] * W[k * n + i] for i in range(n)])
                try:
                    V[vi] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                except:
                    pass

        for k in range(p):
            for i in range(n):
                wi = k * n + i
                A = np.array([U[i * m + j] * V[j * p + k] for j in range(m)])
                try:
                    W[wi] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                except:
                    pass

        if it % 20 == 0:
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                        err += (s - 1.0)**2
            if err < 1e-12:
                break

    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                err += (s - 1.0)**2

    return U, V, W, err

def _try_round(U, V, W, n, m, p):
    for scale in [1, 2, 4, 8, 16, 32, 64]:
        Ur = np.round(U * scale) / scale
        Vr = np.round(V * scale) / scale
        Wr = np.round(W * scale) / scale

        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Ur[i * m + j], Vr[j * p + k] * Wr[k * n + i])
                    if abs(s - 1.0) > 1e-8:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break

        if ok and _verify_exact(n, m, p, Ur.tolist(), Vr.tolist(), Wr.tolist()):
            return {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}

    return None

def _construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)

    for restart in range(20):
        if time.time() > _DEADLINE - 25:
            break

        U, V, W, err = _als(n, m, p, 49, seed=42 + restart * 17, max_iter=100)

        if err < 1e-8:
            result = _try_round(U, V, W, n, m, p)
            if result is not None:
                return result

    return best

def construct():
    try:
        return _construct()
    except Exception:
        n, m, p = DIMENSIONS
        return _schoolbook(n, m, p)
