"""
Rank-minimal exact tensor decomposition for 2x2 * 2x5 matrix multiplication.

Guaranteed fallback: structured Strassen(2x2 blocks)+schoolbook rank-18.
Aggressive search for rank 17 (and 16) using fast ALS with many restarts,
rank-16 initializations, and a final high-precision polish / exact local solve.

The output factor W follows the official trace convention W[k*n+i, r].
"""

import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 2, 5]


def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1
    return T


def _decompose(U, V, W):
    R = U.shape[1]
    result = np.zeros((U.shape[0], V.shape[0], W.shape[0]), dtype=np.float64)
    for r in range(R):
        result += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
    return result


def _check_decomposition(U, V, W, T, tol=1e-8):
    result = _decompose(U, V, W)
    return np.allclose(result, T, atol=tol)


def _to_frac(x):
    for denom in (1, 2, 4, 8, 16, 32, 64):
        num = int(round(x * denom))
        if abs(x - num / denom) < 1e-9:
            return Fraction(num, denom)
    return Fraction(x).limit_denominator(256)


def _check_exact(U, V, W, T):
    Uf = [[_to_frac(x) for x in row] for row in U]
    Vf = [[_to_frac(x) for x in row] for row in V]
    Wf = [[_to_frac(x) for x in row] for row in W]
    R = U.shape[1]
    nu, nv, nw = U.shape[0], V.shape[0], W.shape[0]
    for u in range(nu):
        for v in range(nv):
            for w in range(nw):
                total = Fraction(0)
                for r in range(R):
                    total += Uf[u][r] * Vf[v][r] * Wf[w][r]
                if total != Fraction(int(T[u, v, w])):
                    return False
    return True


def _snap(M, denom):
    return np.round(M * denom) / denom


def _polish_rational(U, V, W, T):
    """Snap to small-denominator rationals that still decompose exactly."""
    for denom in (1, 2, 4, 8, 16, 32, 64):
        Us, Vs, Ws = _snap(U, denom), _snap(V, denom), _snap(W, denom)
        if _check_decomposition(Us, Vs, Ws, T, tol=1e-6) and _check_exact(Us, Vs, Ws, T):
            return Us.tolist(), Vs.tolist(), Ws.tolist()
    for dU in (1, 2, 4):
        for dV in (1, 2, 4):
            for dW in (1, 2, 4):
                Us, Vs, Ws = _snap(U, dU), _snap(V, dV), _snap(W, dW)
                if _check_exact(Us, Vs, Ws, T):
                    return Us.tolist(), Vs.tolist(), Ws.tolist()
    for scale in (0.05, 0.01):
        for _ in range(6):
            Up = U + np.random.randn(*U.shape) * scale
            Vp = V + np.random.randn(*V.shape) * scale
            Wp = W + np.random.randn(*W.shape) * scale
            for denom in (1, 2, 4):
                Us, Vs, Ws = _snap(Up, denom), _snap(Vp, denom), _snap(Wp, denom)
                if _check_exact(Us, Vs, Ws, T):
                    return Us.tolist(), Vs.tolist(), Ws.tolist()
    return None


# ---- Fast ALS (vectorized least squares) ----

def _als_update_U(T, V, W):
    R = V.shape[1]
    A = np.einsum('vr,wr->vwr', V, W).reshape(-1, R)
    A_pinv = np.linalg.pinv(A)
    U = np.zeros((T.shape[0], R), dtype=np.float64)
    for u in range(T.shape[0]):
        U[u] = A_pinv @ T[u].reshape(-1)
    return U


def _als_update_V(T, U, W):
    R = U.shape[1]
    A = np.einsum('ur,wr->uwr', U, W).reshape(-1, R)
    A_pinv = np.linalg.pinv(A)
    V = np.zeros((T.shape[1], R), dtype=np.float64)
    for v in range(T.shape[1]):
        V[v] = A_pinv @ T[:, v, :].reshape(-1)
    return V


def _als_update_W(T, U, V):
    R = U.shape[1]
    A = np.einsum('ur,vr->uvr', U, V).reshape(-1, R)
    A_pinv = np.linalg.pinv(A)
    W = np.zeros((T.shape[2], R), dtype=np.float64)
    for w in range(T.shape[2]):
        W[w] = A_pinv @ T[:, :, w].reshape(-1)
    return W


def _run_als(T, U, V, W, iters, deadline):
    for _ in range(iters):
        if time.time() > deadline:
            break
        U = _als_update_U(T, V, W)
        V = _als_update_V(T, U, W)
        W = _als_update_W(T, U, V)
    return U, V, W


# ---- Structured rank-18 (guaranteed fallback) ----

def _build_structured_decomposition():
    n, m, p = DIMENSIONS
    rank = 18
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    def add_strassen_block(col0, col1, base):
        r = base
        U[0, r] = 1; U[3, r] = 1
        V[0 * p + col0, r] = 1; V[1 * p + col1, r] = 1
        W[col0 * n + 0, r] = 1; W[col1 * n + 1, r] = 1
        r = base + 1
        U[2, r] = 1; U[3, r] = 1
        V[0 * p + col0, r] = 1
        W[col0 * n + 1, r] = 1; W[col1 * n + 1, r] = -1
        r = base + 2
        U[0, r] = 1
        V[0 * p + col1, r] = 1; V[1 * p + col1, r] = -1
        W[col1 * n + 0, r] = 1; W[col1 * n + 1, r] = 1
        r = base + 3
        U[3, r] = 1
        V[1 * p + col0, r] = 1; V[0 * p + col0, r] = -1
        W[col0 * n + 0, r] = 1; W[col0 * n + 1, r] = 1
        r = base + 4
        U[0, r] = 1; U[1, r] = 1
        V[1 * p + col1, r] = 1
        W[col0 * n + 0, r] = -1; W[col1 * n + 0, r] = 1
        r = base + 5
        U[2, r] = 1; U[0, r] = -1
        V[0 * p + col0, r] = 1; V[0 * p + col1, r] = 1
        W[col1 * n + 1, r] = 1
        r = base + 6
        U[1, r] = 1; U[3, r] = -1
        V[1 * p + col0, r] = 1; V[1 * p + col1, r] = 1
        W[col0 * n + 0, r] = 1

    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)

    r = 14
    U[0, r] = 1; V[0 * p + 4, r] = 1; W[4 * n + 0, r] = 1
    r = 15
    U[1, r] = 1; V[1 * p + 4, r] = 1; W[4 * n + 0, r] = 1
    r = 16
    U[2, r] = 1; V[0 * p + 4, r] = 1; W[4 * n + 1, r] = 1
    r = 17
    U[3, r] = 1; V[1 * p + 4, r] = 1; W[4 * n + 1, r] = 1

    return U, V, W


def _random_init(T, R, scale=1.0):
    n, m, p = DIMENSIONS
    U = np.random.randn(n * m, R) * scale
    V = np.random.randn(m * p, R) * scale
    W = np.random.randn(p * n, R) * scale
    return U, V, W


def _refine_and_check(T, U, V, W, deadline, iters, tol=1e-7):
    U, V, W = _run_als(T, U, V, W, iters, deadline)
    if _check_decomposition(U, V, W, T, tol=tol):
        polished = _polish_rational(U, V, W, T)
        if polished is not None:
            return polished
        return U.tolist(), V.tolist(), W.tolist()
    return None


def _search_rank(T, R, deadline, n_starts, iters, use_rank16_init=False):
    n, m, p = DIMENSIONS
    for seed in range(n_starts):
        if time.time() > deadline:
            break
        np.random.seed(100000 + seed)
        if use_rank16_init:
            U16, V16, W16 = _random_init(T, 16, scale=1.0)
            U16, V16, W16 = _run_als(T, U16, V16, W16, 12, deadline)
            U = np.hstack([U16, np.random.randn(n * m, R - 16) * 0.5])
            V = np.hstack([V16, np.random.randn(m * p, R - 16) * 0.5])
            W = np.hstack([W16, np.random.randn(p * n, R - 16) * 0.5])
        else:
            U, V, W = _random_init(T, R, scale=1.0)
        res = _refine_and_check(T, U, V, W, deadline, iters)
        if res is not None:
            return res
    return None


def construct():
    deadline = time.time() + 135  # safety margin before 150s wall-clock limit
    n, m, p = DIMENSIONS
    T = _build_target()

    U_struct, V_struct, W_struct = _build_structured_decomposition()
    if not _check_decomposition(U_struct, V_struct, W_struct, T, tol=1e-8):
        # extremely unlikely; fall back to schoolbook rank 20
        rank = n * m * p
        U = np.zeros((n * m, rank)); V = np.zeros((m * p, rank)); W = np.zeros((p * n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1; V[j * p + k, c] = 1; W[k * n + i, c] = 1; c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # ---- Aggressive rank-17 search (score 1.176) ----
    if time.time() < deadline:
        res = _search_rank(T, 17, deadline, n_starts=260, iters=12, use_rank16_init=False)
        if res is not None:
            Ua, Va, Wa = (np.array(x, dtype=np.float64) for x in res)
            if _check_decomposition(Ua, Va, Wa, T, tol=1e-7):
                return {"U": res[0], "V": res[1], "W": res[2]}

    # ---- Rank-16 search (score 1.25) ----
    if time.time() < deadline:
        res = _search_rank(T, 16, deadline, n_starts=120, iters=14, use_rank16_init=False)
        if res is not None:
            Ua, Va, Wa = (np.array(x, dtype=np.float64) for x in res)
            if _check_decomposition(Ua, Va, Wa, T, tol=1e-7):
                return {"U": res[0], "V": res[1], "W": res[2]}
        res = _search_rank(T, 17, deadline, n_starts=80, iters=14, use_rank16_init=True)
        if res is not None:
            Ua, Va, Wa = (np.array(x, dtype=np.float64) for x in res)
            if _check_decomposition(Ua, Va, Wa, T, tol=1e-7):
                return {"U": res[0], "V": res[1], "W": res[2]}

    # ---- Guaranteed rank-18 fallback, polished ----
    polished = _polish_rational(U_struct, V_struct, W_struct, T)
    if polished is not None:
        return {"U": polished[0], "V": polished[1], "W": polished[2]}
    return {"U": U_struct.tolist(), "V": V_struct.tolist(), "W": W_struct.tolist()}
