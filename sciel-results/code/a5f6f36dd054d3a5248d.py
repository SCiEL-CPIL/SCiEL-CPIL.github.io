"""Matrix multiplication 3x3 x 3x4: rank-minimal exact tensor decomposition.

We search for a decomposition of the MM(3,3,4) tensor of minimal rank.
The tensor has entries T[(i,j),(j,k),(k,i)] = 1 (trace convention W[k*n+i]).
A rank-r decomposition is U (n*m x r), V (m*p x r), W (p*n x r) with
sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for every (i,j,k).
Entries must be integer or half-integer.
"""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 4]

_DEADLINE = time.time() + 110.0   # safety margin under the 120 s limit


def _within_budget():
    return time.time() < _DEADLINE


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check the tensor identity and integer/half-integer entries."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    if V.shape != (m * p, r) or W.shape != (p * n, r):
        return False
    # integer / half-integer check
    for M in (U, V, W):
        doubled = 2.0 * M
        if not np.allclose(doubled, np.round(doubled), atol=tol):
            return False
    # tensor identity
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _round_to_half(M):
    return np.round(2.0 * M) / 2.0


def _try_construct_fft_like(n, m, p, r, seed=0):
    """
    Try to build a rank-r decomposition using a structured (FFT/Winograd-like)
    basis. We choose U and V with entries from a small half-integer set and
    solve for W = ... Actually we need the full tensor identity, which is
    nonlinear in W. Instead we use a linear approach: fix U and V, then the
    condition is linear in W for each (i,j,k) only if U*V is nonzero per term,
    which it generally is not. So we use a randomized local search instead.
    """
    return None


def _random_search(n, m, p, r, iters, seed=0):
    """
    Randomized search: initialize U,V,W with small half-integer entries and
    iteratively adjust to satisfy the tensor identity. This is heuristic.
    """
    rng = np.random.default_rng(seed)
    best = None
    best_err = None
    for it in range(iters):
        if not _within_budget():
            break
        # initialize with a few nonzero half-integers
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        # sprinkle ones
        for c in range(r):
            idx_u = rng.integers(0, n * m)
            idx_v = rng.integers(0, m * p)
            idx_w = rng.integers(0, p * n)
            U[idx_u, c] = 1
            V[idx_v, c] = 1
            W[idx_w, c] = 1
        # compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] for c in range(r))
                    err += (s - 1.0)**2
        if best_err is None or err < best_err:
            best_err = err
            best = (U.copy(), V.copy(), W.copy())
        if best_err < 1e-8:
            break
    if best is not None and best_err is not None and best_err < 1e-6:
        U, V, W = best
        U = _round_to_half(U); V = _round_to_half(V); W = _round_to_half(W)
        if _verify(U, V, W, n, m, p):
            return U, V, W
    return None


def construct():
    n, m, p = DIMENSIONS
    # Baseline: schoolbook rank 36.
    U0, V0, W0 = _schoolbook(n, m, p)
    best = (36, U0, V0, W0)

    # Try to improve by attempting lower ranks via structured search.
    # Known lower bound for MM(3,3,4) is at least 18 (from Strassen-style).
    # We attempt ranks in increasing order, spending limited time each.
    for r in range(18, 36):
        if not _within_budget():
            break
        # time budget per rank
        t0 = time.time()
        res = _random_search(n, m, p, r, iters=400, seed=r)
        if res is not None:
            U, V, W = res
            if r < best[0]:
                best = (r, U, V, W)
                break  # found a valid lower rank; keep going for even lower?
        # also try a couple more seeds if time remains
        if time.time() - t0 < 25 and _within_budget():
            for s in range(3):
                res = _random_search(n, m, p, r, iters=300, seed=1000 + r * 10 + s)
                if res is not None:
                    U, V, W = res
                    if r < best[0]:
                        best = (r, U, V, W)
                        break
        if best[0] == r:
            break

    rank, U, V, W = best
    # Convert to lists of Python floats (half-integers are exactly representable).
    U = [[float(x) for x in row] for row in np.asarray(U)]
    V = [[float(x) for x in row] for row in np.asarray(V)]
    W = [[float(x) for x in row] for row in np.asarray(W)]
    return {"U": U, "V": V, "W": W}
