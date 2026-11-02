"""Search-based rank-minimal tensor decomposition for 2x3 by 3x4 matmul."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 4]
N, M, P = DIMENSIONS
RANK_TARGET = 12  # theoretical lower bound = max(n*m, m*p, p*n)
DEADLINE = 140.0  # wall-clock safety margin (< 150s kill)
RNG = np.random.default_rng(12345)


def _tensor(U, V, W):
    """Reconstruct the multiplication tensor T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    T = np.zeros((N, M, P))
    for c in range(r):
        T += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
    return T


def _residual(U, V, W):
    """Frobenius norm of (T - T_true)."""
    T = _tensor(U, V, W)
    Ttrue = np.arange(N * M * P, dtype=float).reshape(N, M, P)
    return np.linalg.norm(T - Ttrue)


def _snap_half(X, tol=1e-5):
    """Snap entries to nearest multiple of 0.5 if within tol."""
    Y = (2.0 * X).round() / 2.0
    if np.max(np.abs(Y - X)) <= tol:
        return Y
    return None


def _is_valid(U, V, W):
    """Check exact decomposition with half-integer entries."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False
    for X in (U, V, W):
        if not np.allclose(2.0 * X, np.round(2.0 * X), atol=1e-7):
            return False
    T = _tensor(U, V, W)
    Ttrue = np.arange(N * M * P, dtype=float).reshape(N, M, P)
    return np.max(np.abs(T - Ttrue)) <= 1e-7


def _refine(U, V, W, steps=6000, lr0=0.15, tol=1e-9):
    """Gradient-descent refinement of the factor matrices (smooth min of residual^2)."""
    U = U.astype(float).copy()
    V = V.astype(float).copy()
    W = W.astype(float).copy()
    Ttrue = np.arange(N * M * P, dtype=float).reshape(N, M, P)
    r = U.shape[1]
    lr = lr0
    for it in range(steps):
        T = _tensor(U, V, W)
        dT = T - Ttrue
        res = np.linalg.norm(dT)
        if res < tol:
            break
        gU = np.einsum('abc,bc->ac', dT, V * W)
        gV = np.einsum('abc,ac->bc', dT, U * W)
        gW = np.einsum('abc,ab->bc', dT, U * V)
        # adaptive step (simple backtracking-lite)
        lr = min(0.5, max(1e-6, lr0 / (1.0 + 0.02 * it)))
        U -= lr * gU
        V -= lr * gV
        W -= lr * gW
    return U, V, W, _residual(U, V, W)


def _try_rank(rank, time_budget):
    """Random reparametrization search + refinement for a given rank."""
    best = None
    while time.time() < time_budget:
        U = RNG.standard_normal((N * M, rank)) * 0.6
        V = RNG.standard_normal((M * P, rank)) * 0.6
        W = RNG.standard_normal((P * N, rank)) * 0.6
        U, V, W, res = _refine(U, V, W, steps=8000, lr0=0.2)
        if best is None or res < best[0]:
            best = (res, U.copy(), V.copy(), W.copy())
        if best[0] < 1e-8:
            break
    if best is None:
        return None
    res, U, V, W = best
    # high-precision polish
    U, V, W, res2 = _refine(U, V, W, steps=12000, lr0=0.05, tol=1e-12)
    if res2 < res:
        U, V, W, res = U, V, W, res2
    # snap to half-integers
    Us, Vs, Ws = _snap_half(U), _snap_half(V), _snap_half(W)
    if Us is not None and Vs is not None and Ws is not None and _is_valid(Us, Vs, Ws):
        return (Us.tolist(), Vs.tolist(), Ws.tolist())
    # if exact within tight tol even without snap, try rounding harder
    if res < 1e-6:
        Ur = (2 * U).round() / 2
        Vr = (2 * V).round() / 2
        Wr = (2 * W).round() / 2
        if _is_valid(Ur, Vr, Wr):
            return (Ur.tolist(), Vr.tolist(), Wr.tolist())
    return None


def _schoolbook():
    n, m, p = N, M, P
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


def construct():
    start = time.time()
    # Try the theoretical-minimum rank first, then larger ranks as fallback.
    for rank in (RANK_TARGET, 13, 14, 15, 16, 18, 20, 24):
        if time.time() - start > DEADLINE - 20.0:
            break
        res = _try_rank(rank, time.time() + min(25.0, DEADLINE - (time.time() - start)))
        if res is not None:
            U, V, W = res
            return {"U": U, "V": V, "W": W}
    # Fallback: guaranteed-valid schoolbook rank 24.
    return _schoolbook()
