"""Search for a low-rank exact tensor decomposition of 5x5x5 matrix multiplication.

Strategy: alternating least squares (ALS) on the tensor
    T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]  = delta_{i,k}
for decreasing target ranks, then round to half-integers and verify exactness.
A guaranteed schoolbook (rank 125) decomposition is the fallback, and a wall-clock
deadline guard ensures we return the best valid result before the 150 s limit.
"""

import time
import numpy as np

DIMENSIONS = [5, 5, 5]

_START = time.time()
# Safety margin well before the 150 s hard kill.
DEADLINE = 110.0


def _time_left():
    return DEADLINE - (time.time() - _START)


def _schoolbook():
    """Guaranteed-valid rank 125 decomposition (the seed)."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _residual(U, V, W):
    """R[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] - delta_{i,k}."""
    n, m, p = DIMENSIONS
    Ur = np.asarray(U, dtype=float).reshape(n, m, -1)
    Vr = np.asarray(V, dtype=float).reshape(m, p, -1)
    Wr = np.asarray(W, dtype=float).reshape(p, n, -1)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    target = np.zeros((n, m, p))
    for i in range(n):
        target[i, :, i] = 1.0
    return T - target


def _is_valid(U, V, W, tol=1e-9):
    return bool(np.all(np.abs(_residual(U, V, W)) <= tol))


def _round_half(X):
    return (2.0 * np.asarray(X, dtype=float)).round() / 2.0


def _clean(X):
    """Convert integer-valued floats to ints; keep true half-integers as floats."""
    out = []
    for row in X:
        nr = []
        for v in row:
            vf = float(v)
            if abs(vf - round(vf)) < 1e-12:
                nr.append(int(round(vf)))
            else:
                nr.append(vf)
        out.append(nr)
    return out


def _als(rank, iters, seed):
    """Alternating least squares for a fixed rank; each factor row is a small linear solve."""
    rng = np.random.default_rng(seed)
    n, m, p = DIMENSIONS
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    # Initialize from a random subset of schoolbook columns to inject structure.
    cols = rng.permutation(n * m * p)[:rank]
    for c in range(rank):
        s = cols[c]
        i = s // (m * p)
        j = (s // p) % m
        k = s % p
        U[i * m + j, c] = 1.0
        V[j * p + k, c] = 1.0
        W[k * n + i, c] = 1.0

    for _ in range(iters):
        if _time_left() < 3.0:
            break
        # Update U: for each (i,j), solve p x rank system, b[k]=delta_{i,k}.
        for i in range(n):
            for j in range(m):
                A = V[j * p:(j + 1) * p, :] * W[i::n, :].T
                b = np.zeros(p)
                b[i] = 1.0
                U[i * m + j, :], *_ = np.linalg.lstsq(A, b, rcond=None)
        # Update V: for each (j,k), solve n x rank system, b[i]=delta_{i,k}.
        for j in range(m):
            for k in range(p):
                A = U[j::m, :] * W[k::n, :].T
                b = np.zeros(n)
                b[k] = 1.0
                V[j * p + k, :], *_ = np.linalg.lstsq(A, b, rcond=None)
        # Update W: for each (k,i), solve m x rank system, b[j]=delta_{i,k}.
        for k in range(p):
            for i in range(n):
                A = U[i * m:(i + 1) * m, :] * V[k::p, :].T
                b = np.zeros(m)
                b[i] = 1.0
                W[k * n + i, :], *_ = np.linalg.lstsq(A, b, rcond=None)
    return U, V, W


def _local_refine(U, V, W, steps=1500):
    """Smooth L2-surrogate coordinate search over half-integer flips; keep best."""
    best = (U.copy(), V.copy(), W.copy())
    best_err = float(np.sum(_residual(*best) ** 2))
    mats = [U, V, W]
    for _ in range(steps):
        if _time_left() < 3.0:
            break
        X = np.random.choice(mats)
        r = np.random.randint(X.shape[0])
        c = np.random.randint(X.shape[1])
        X[r, c] += np.random.choice([-0.5, 0.5])
        err = float(np.sum(_residual(U, V, W) ** 2))
        if err < best_err:
            best_err = err
            best = (U.copy(), V.copy(), W.copy())
        else:
            U, V, W = best
    return best


def construct():
    # Guaranteed fallback (never score below the seed).
    fallback = _schoolbook()
    try:
        for rank in [49, 55, 70, 95]:
            if _time_left() < 6.0:
                break
            for seed in range(3):
                if _time_left() < 4.0:
                    break
                try:
                    U, V, W = _als(rank, 300, seed)
                    Ur, Vr, Wr = _round_half(U), _round_half(V), _round_half(W)
                    if _is_valid(Ur, Vr, Wr):
                        return {"U": _clean(Ur), "V": _clean(Vr), "W": _clean(Wr)}
                    Ur, Vr, Wr = _local_refine(Ur, Vr, Wr)
                    if _is_valid(Ur, Vr, Wr):
                        return {"U": _clean(Ur), "V": _clean(Vr), "W": _clean(Wr)}
                except Exception:
                    continue
    except Exception:
        pass
    return fallback
