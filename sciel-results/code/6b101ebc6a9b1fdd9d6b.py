"""Randomized exact rank search for 2x3 x 3x3 matrix multiplication."""
import random
import time
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
_DEADLINE = 150.0
_START = None


def _elapsed():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START


def _deadline_exceeded(margin=5.0):
    return _elapsed() >= _DEADLINE - margin


def _verify(U, V, W, n, m, p, tol=1e-9):
    r = U.shape[1]
    T = np.zeros((n, m, p))
    for c in range(r):
        uc = U[:, c].reshape(n, m)
        vc = V[:, c].reshape(m, p)
        wc = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
    return np.max(np.abs(T - 1.0)) < tol


def _solve_for_W(U, V, n, m, p):
    r = U.shape[1]
    W = np.zeros((p * n, r))
    for k in range(p):
        A = np.zeros((n * m, r))
        for i in range(n):
            for j in range(m):
                idx = i * m + j
                for c in range(r):
                    A[idx, c] = U[idx, c] * V[j * p + k, c]
        b = np.ones(n * m)
        try:
            if r == n * m:
                # Square system
                x = np.linalg.solve(A, b)
            else:
                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except (np.linalg.LinAlgError, np.linalg.LstsqWarning):
            return None
        W[k * n:(k + 1) * n, :] = x.reshape(n, r)
    return W


def _round_halfint(W):
    return np.round(W * 2.0) / 2.0


def _is_halfint(arr, tol=1e-9):
    return np.max(np.abs(arr * 2.0 - np.round(arr * 2.0))) < tol


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _random_UV(n, m, p, r, rng, scale=2.0):
    vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, scale])
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    for c in range(r):
        U[:, c] = rng.choice(vals, size=n * m)
        V[:, c] = rng.choice(vals, size=m * p)
    return U, V


def construct():
    n, m, p = DIMENSIONS
    global _START
    _START = time.time()

    best = None

    for r in range(6, 19):
        if _deadline_exceeded():
            break
        for seed in range(50):
            if _deadline_exceeded():
                break
            rng = random.Random(12345 + r * 1000 + seed)
            max_iters = 8000
            for it in range(max_iters):
                if _deadline_exceeded():
                    break
                U, V = _random_UV(n, m, p, r, rng, scale=2.0)
                W = _solve_for_W(U, V, n, m, p)
                if W is None:
                    continue
                Wr = _round_halfint(W)
                if _is_halfint(Wr) and _verify(U, V, Wr, n, m, p):
                    best = (U, V, Wr)
                    break
            if best is not None:
                break
        if best is not None:
            break

    if best is None:
        U, V, W = _schoolbook(n, m, p)
    else:
        U, V, W = best

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
