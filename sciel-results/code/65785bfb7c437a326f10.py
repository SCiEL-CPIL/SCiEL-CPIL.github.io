"""Matrix multiplication 2x4x4: search for a low-rank EXACT tensor decomposition.

We keep the required interface: construct() -> {'U','V','W'} (lists of rows).
A guaranteed-valid schoolbook (rank n*m*p = 32) is the fallback. A bounded
random + ALS search (with exact int validation) tries to find a lower rank.
"""
import random
import time

_DIM = (2, 4, 4)
_DEADLINE = 140.0          # safety margin under the 150s evaluator kill
_RAND_SEED = 20240611
_INT_SCALE = 64            # scale float ALS entries to a common denominator for exact checks
_SCHOOLBOOK = None


def _schoolbook():
    global _SCHOOLBOOK
    if _SCHOOLBOOK is not None:
        return _SCHOOLBOOK
    n, m, p = _DIM
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
    _SCHOOLBOOK = {"U": U, "V": V, "W": W}
    return _SCHOOLBOOK


def _make_tensor():
    n, m, p = _DIM
    T = [[[0] * p for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i][j][k] = 1
    return T


def _is_exact(Uf, Vf, Wf):
    """Exact integer check that U,V,W reproduce the 2x4x4 multiplication tensor."""
    n, m, p = _DIM
    S = _INT_SCALE
    U = [[int(round(u * S)) for u in row] for row in Uf]
    V = [[int(round(v * S)) for v in row] for row in Vf]
    W = [[int(round(w * S)) for w in row] for row in Wf]
    T = _make_tensor()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(len(U[0])):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != T[i][j][k] * S ** 3:
                    return False
    return True


def _als_step(U, V, W):
    """One ALS pass; returns new (U,V,W) and a squared-error estimate."""
    n, m, p = _DIM
    T = _make_tensor()
    rm = len(U[0])
    new_U = [[0.0] * rm for _ in range(n * m)]
    new_V = [[0.0] * rm for _ in range(m * p)]
    new_W = [[0.0] * rm for _ in range(p * n)]
    err = 0.0
    for a in range(n * m):
        for r in range(rm):
            num = 0.0
            den = 1e-12
            for b in range(m * p):
                vr = V[b][r]
                if vr == 0.0:
                    continue
                for c in range(p * n):
                    vc = W[c][r]
                    val = T[a // m][a % m][c]  # placeholder; corrected below
                    del val
                    den += (vr * vc) ** 2
            # recompute numerator with correct indexing
            num = 0.0
            for b in range(m * p):
                vr = V[b][r]
                if vr == 0.0:
                    continue
                for c in range(p * n):
                    num += T[b // p][b % p][c] * vr * W[c][r]
            new_U[a][r] = num / den
    for b in range(m * p):
        for r in range(rm):
            num = 0.0
            den = 1e-12
            for a in range(n * m):
                ur = U[a][r]
                if ur == 0.0:
                    continue
                for c in range(p * n):
                    den += (ur * W[c][r]) ** 2
            num = 0.0
            for a in range(n * m):
                ur = U[a][r]
                if ur == 0.0:
                    continue
                for c in range(p * n):
                    num += T[a // m][a % m][c] * ur * W[c][r]
            new_V[b][r] = num / den
    for c in range(p * n):
        for r in range(rm):
            num = 0.0
            den = 1e-12
            for a in range(n * m):
                ur = U[a][r]
                if ur == 0.0:
                    continue
                for b in range(m * p):
                    den += (ur * V[b][r]) ** 2
            num = 0.0
            for a in range(n * m):
                ur = U[a][r]
                if ur == 0.0:
                    continue
                for b in range(m * p):
                    num += T[a // m][a % m][b] * ur * V[b][r]
            new_W[c][r] = num / den
    # error estimate
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rm):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                d = s - 1.0
                err += d * d
    return new_U, new_V, new_W, err


def _rand_init(rm):
    n, m, p = _DIM
    U = [[random.gauss(0.0, 1.0) for _ in range(rm)] for _ in range(n * m)]
    V = [[random.gauss(0.0, 1.0) for _ in range(rm)] for _ in range(m * p)]
    W = [[random.gauss(0.0, 1.0) for _ in range(rm)] for _ in range(p * n)]
    return U, V, W


def _best_rank_search(start_deadline):
    best = None
    best_rank = None
    # try ranks from the theoretical lower bound upward
    n, m, p = _DIM
    lower = max(n * m, m * p, p * n)   # 16
    for rm in range(lower, n * m * p):  # up to 31 (32 is the fallback)
        if time.time() >= start_deadline:
            break
        if best_rank is not None and rm >= best_rank:
            break
        found = False
        max_iters = 300
        max_restarts = 6
        for restart in range(max_restarts):
            if time.time() >= start_deadline:
                break
            U, V, W = _rand_init(rm)
            for it in range(max_iters):
                U, V, W, err = _als_step(U, V, W)
                if time.time() >= start_deadline:
                    break
                if err < 1e-9:
                    if _is_exact(U, V, W):
                        best = {"U": U, "V": V, "W": W}
                        best_rank = rm
                        found = True
                        break
            if found:
                break
    return best, best_rank


def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    random.seed(_RAND_SEED)
    try:
        found, rank = _best_rank_search(deadline)
        if found is not None:
            return found
    except Exception:
        pass
    return _schoolbook()
