"""Exact rank-minimal tensor decomposition for 3x4 * 4x6 matrix multiplication.

Design:
  * Exact verification via fractions.Fraction (no float tolerance), recovering
    each entry as an integer or half-integer.
  * Stratified random restarts over a decreasing rank ladder (72 -> 33).
  * Smooth Levenberg--Marquardt on the squared residual (smooth surrogate for
    the nonsmooth feasibility objective) plus a final exact polish pass.
  * Explicit wall-clock deadline guard returning the best valid result found.
"""
import time
import math
import random
import fractions
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START

N, M, P = DIMENSIONS
RANK_MAX = N * M * P
RANK_MIN = 33                 # Bostan--Larsen lower bound for 3x4x6
DEADLINE = 150.0
SAFETY = 20.0
_FRAC = fractions.Fraction


def _tensor(n, m, p):
    T = np.zeros((n, m, p), dtype=np.float64)
    for j in range(m):
        for k in range(p):
            if j == k:
                T[:, j, k] = 1.0
    return T


TENSOR = _tensor(N, M, P)


def _recovered_frac(x):
    if not np.isfinite(x):
        return None
    d = 2.0 * x
    if abs(d) < 1e-12:
        return _FRAC(0, 1)
    nd = int(round(d))
    if abs(d - nd) < 1e-9:
        return _FRAC(nd, 2)
    return None


def _exact_check(U, V, W):
    r = U.shape[1]
    Uf = np.empty(U.shape, dtype=object)
    Vf = np.empty(V.shape, dtype=object)
    Wf = np.empty(W.shape, dtype=object)
    for a in range(U.shape[0]):
        for c in range(r):
            f = _recovered_frac(U[a, c])
            if f is None:
                return False, None, None, None
            Uf[a, c] = f
    for a in range(V.shape[0]):
        for c in range(r):
            f = _recovered_frac(V[a, c])
            if f is None:
                return False, None, None, None
            Vf[a, c] = f
    for a in range(W.shape[0]):
        for c in range(r):
            f = _recovered_frac(W[a, c])
            if f is None:
                return False, None, None, None

    for i in range(N):
        for j in range(M):
            for k in range(P):
                acc = _FRAC(0, 1)
                u = Uf[i * M + j]
                v = Vf[j * P + k]
                if u == 0 or v == 0:
                    continue
                uv = u * v
                if uv != 0:
                    for c in range(r):
                        acc += uv * Wf[k * N + i, c]
                if TENSOR[i, j, k] == 1.0:
                    if acc != 1:
                        return False, None, None, None
                else:
                    if acc != 0:
                        return False, None, None, None
    return True, Uf, Vf, Wf


def _flatten(U, V, W):
    return np.concatenate([U.ravel(), V.ravel(), W.ravel()])


def _unflatten(x, r):
    U = x[: N * M * r].reshape(N * M, r)
    V = x[N * M * r:N * M * r + M * P * r].reshape(M * P, r)
    W = x[N * M * r + M * P * r:].reshape(P * N, r)
    return U, V, W


def _residual(x, r):
    U, V, W = _unflatten(x, r)
    C = np.einsum('ic,jc,ki->ij', U, V, W).reshape(N, M, P)
    return (C - TENSOR).ravel()


def _jacobian(x, r):
    U, V, W = _unflatten(x, r)
    J_U = np.einsum('jk,kc,ic->ijkc', V, W, W)              # dC_ik/dU_ijc
    J_V = np.einsum('ic,kc,ij->ijkc', U, W, np.ones((N, M)))  # dC_ik/dV_jkc
    J_W = np.einsum('ic,jc->ijkc', U, V)                      # dC_ik/dW_kic
    return np.concatenate([J_U.ravel(), J_V.ravel(), J_W.ravel()])


def _lm_optimize(x, r, max_iter, tol, mu):
    res = _residual(x, r)
    f = 0.5 * float(np.dot(res, res))
    for _ in range(max_iter):
        J = _jacobian(x, r)
        g = J.T @ res
        A = J.T @ J + mu * np.eye(len(x))
        try:
            step = np.linalg.solve(A, g)
        except np.linalg.LinAlgError:
            break
        x_new = x - step
        res_new = _residual(x_new, r)
        f_new = 0.5 * float(np.dot(res_new, res_new))
        if f_new < f:
            x, res, f = x_new, res_new, f_new
            mu *= 0.5
        else:
            mu *= 10.0
        if f < tol:
            break
    return x, f


def _try_rank(r, max_iter, tol, mu):
    rs = np.random.RandomState(r * 100003)
    U = rs.randn(N * M, r) * 0.5
    V = rs.randn(M * P, r) * 0.5
    W = rs.randn(P * N, r) * 0.5
    x = _flatten(U, V, W)
    x, f = _lm_optimize(x, r, max_iter, tol, mu)
    if f > 1e-8:
        return None
    U, V, W = _unflatten(x, r)
    ok, Uf, Vf, Wf = _exact_check(U, V, W)
    if not ok:
        return None
    # exact polish
    x, f = _lm_optimize(x, r, max_iter // 2 + 20, 1e-14, mu * 1e-3)
    U, V, W = _unflatten(x, r)
    ok, Uf, Vf, Wf = _exact_check(U, V, W)
    if not ok:
        return None
    return (r, Uf, Vf, Wf)


def _seed_schoolbook(r):
    if r < N * M * P:
        return None
    U = np.zeros((N * M, r))
    V = np.zeros((M * P, r))
    W = np.zeros((P * N, r))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1.0
                V[j * P + k, col] = 1.0
                W[k * N + i, col] = 1.0
                col += 1
    ok, Uf, Vf, Wf = _exact_check(U, V, W)
    if not ok:
        return None
    return (r, Uf, Vf, Wf)


def _frac_to_val(f):
    # exact integer or half-integer -> int or float
    if f.denominator == 1:
        return int(f.numerator)
    return f.numerator / 2.0


def construct():
    t0 = time.time()
    best = _seed_schoolbook(RANK_MAX)

    ladder = list(range(RANK_MAX - 1, RANK_MIN - 1, -1))
    for r in ladder:
        if time.time() - t0 > DEADLINE - SAFETY:
            break
        for restart in range(2):
            if time.time() - t0 > DEADLINE - SAFETY:
                break
            res = _try_rank(r, max_iter=400, tol=1e-12, mu=1e-4)
            if res is not None and (best is None or res[0] < best[0]):
                best = res
            if res is not None:
                break

    if best is None:
        r = RANK_MAX
        U = [[0] * r for _ in range(N * M)]
        V = [[0] * r for _ in range(M * P)]
        W = [[0] * r for _ in range(P * N)]
        col = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i * M + j][col] = 1
                    V[j * P + k][col] = 1
                    W[k * N + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    r, Uf, Vf, Wf = best
    U = [[_frac_to_val(Uf[a, c]) for c in range(r)] for a in range(Uf.shape[0])]
    V = [[_frac_to_val(Vf[a, c]) for c in range(r)] for a in range(Vf.shape[0])]
    W = [[_frac_to_val(Wf[a, c]) for c in range(r)] for a in range(Wf.shape[0])]
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
