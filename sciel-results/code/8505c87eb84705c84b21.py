"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]
_START = time.time()
_DEADLINE = 120.0

def _elapsed():
    return time.time() - _START

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _to_halfint(x):
    # round to nearest half-integer
    return int(round(x * 2)) / 2.0

def _exact_verify(U, V, W):
    # U,V,W are lists of half-integers; verify exact decomposition with Fraction
    n, m, p = DIMENSIONS
    rank = len(U[0])
    Uf = [[Fraction(int(round(u*2)))/2 for u in row] for row in U]
    Vf = [[Fraction(int(round(v*2)))/2 for v in row] for row in V]
    Wf = [[Fraction(int(round(w*2)))/2 for w in row] for row in W]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += Uf[i*m+j][r] * Vf[j*p+k][r] * Wf[k*n+i][r]
                if s != 1:
                    return False
    return True

def construct():
    if _elapsed() > _DEADLINE - 10.0:
        return _schoolbook()
    n, m, p = DIMENSIONS
    target = n*m*p
    best = _schoolbook()
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    for rank in range(12, target):
        if _elapsed() > _DEADLINE - 25.0:
            break
        for attempt in range(3):
            if _elapsed() > _DEADLINE - 25.0:
                break
            rng = np.random.default_rng(1000*rank + attempt)
            U = rng.standard_normal((n*m, rank))
            V = rng.standard_normal((m*p, rank))
            W = rng.standard_normal((p*n, rank))
            for it in range(40):
                if _elapsed() > _DEADLINE - 25.0:
                    break
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((rank, rank)); b = np.zeros(rank)
                        for k in range(p):
                            A += np.outer(V[j*p+k, :], W[k*n+i, :])
                            b += T[i*m+j, j*p+k, k*n+i] * W[k*n+i, :]
                        try: U[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except np.linalg.LinAlgError: pass
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((rank, rank)); b = np.zeros(rank)
                        for i in range(n):
                            A += np.outer(U[i*m+j, :], W[k*n+i, :])
                            b += T[i*m+j, j*p+k, k*n+i] * U[i*m+j, :]
                        try: V[j*p+k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except np.linalg.LinAlgError: pass
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((rank, rank)); b = np.zeros(rank)
                        for j in range(m):
                            A += np.outer(U[i*m+j, :], V[j*p+k, :])
                            b += T[i*m+j, j*p+k, k*n+i] * V[j*p+k, :]
                        try: W[k*n+i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except np.linalg.LinAlgError: pass
                # quick float check
                ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = sum(U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] for r in range(rank))
                            if abs(s-1.0) > 1e-6:
                                ok = False; break
                        if not ok: break
                    if not ok: break
                if ok:
                    Uh = [[_to_halfint(x) for x in row] for row in U]
                    Vh = [[_to_halfint(x) for x in row] for row in V]
                    Wh = [[_to_halfint(x) for x in row] for row in W]
                    if _exact_verify(Uh, Vh, Wh):
                        return {"U": Uh, "V": Vh, "W": Wh}
    return best
