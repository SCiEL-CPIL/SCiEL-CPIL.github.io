"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import time

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for rr in range(r):
                    s += Fraction(U[i * m + j][rr]) * Fraction(V[j * p + k][rr]) * Fraction(W[k * n + i][rr])
                if s != 1:
                    return False
    return True

def _to_lists(U, V, W):
    n, m, p = DIMENSIONS
    r = U.shape[2]
    Ul = [[U[i, j, rr] for rr in range(r)] for i in range(n) for j in range(m)]
    Vl = [[V[j, k, rr] for rr in range(r)] for j in range(m) for k in range(p)]
    Wl = [[W[k, i, rr] for rr in range(r)] for k in range(p) for i in range(n)]
    return Ul, Vl, Wl

def construct():
    start = time.time()
    deadline = start + 115.0  # margin before 150s kill
    n, m, p = DIMENSIONS
    T = np.ones((n, m, p))

    def resid(U, V, W):
        R = np.empty((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    R[i, j, k] = np.dot(U[i, j], V[j, k] * W[k, i])
        return float(np.linalg.norm(R - T))

    def als(U, V, W, iters=400):
        for it in range(iters):
            for i in range(n):
                for j in range(m):
                    A = (V[j] * W[:, i]).T
                    b = T[i, j]
                    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                    U[i, j] = sol
            for j in range(m):
                for k in range(p):
                    A = (U[:, j] * W[k, :]).T
                    b = T[:, j, k]
                    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                    V[j, k] = sol
            for k in range(p):
                for i in range(n):
                    A = U[i] * V[:, k]
                    b = T[i, :, k]
                    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                    W[k, i] = sol
            if it % 50 == 0:
                rr = resid(U, V, W)
                if rr < 1e-9:
                    break
        return resid(U, V, W)

    def try_rank(r, restarts=4):
        best = None
        best_res = np.inf
        rng = np.random.default_rng(r * 1000 + 7)
        for s in range(restarts):
            U = rng.normal(size=(n, m, r))
            V = rng.normal(size=(m, p, r))
            W = rng.normal(size=(p, n, r))
            res = als(U, V, W, iters=300)
            if res < best_res:
                best_res = res
                best = (U.copy(), V.copy(), W.copy())
            if time.time() > deadline:
                break
            if best_res < 1e-8:
                break
        return best, best_res

    def round_verify(U, V, W, dens=(1, 2, 4, 8, 16)):
        for d in dens:
            inv = 1.0 / d
            Ul = [[round(U[i, j, rr] * d) * inv for rr in range(U.shape[2])] for i in range(n) for j in range(m)]
            Vl = [[round(V[j, k, rr] * d) * inv for rr in range(V.shape[2])] for j in range(m) for k in range(p)]
            Wl = [[round(W[k, i, rr] * d) * inv for rr in range(W.shape[2])] for k in range(p) for i in range(n)]
            if _verify(Ul, Vl, Wl):
                return Ul, Vl, Wl
        return None

    # search ranks low to high
    ranks = [18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 30, 32, 36, 40, 45, 48, 54, 60, 66, 72]
    for r in ranks:
        if time.time() > deadline:
            break
        best, best_res = try_rank(r, restarts=3)
        if best is not None and best_res < 1e-7:
            rv = round_verify(*best)
            if rv is not None:
                Ul, Vl, Wl = rv
                return {"U": Ul, "V": Vl, "W": Wl}
    return _schoolbook()
# EVOLVE-BLOCK-END
