"""Matrix multiplication 2,2,5: search for a low-rank exact decomposition."""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [2, 2, 5]
START = time.time()
DEADLINE = 150.0
MARGIN = 12.0


def _time_ok():
    return (time.time() - START) < (DEADLINE - MARGIN)


def _residual(T, cols):
    """Residual tensor after removing the given columns (U,V,W each rank x row)."""
    r = T.copy()
    for c in range(len(cols)):
        r -= np.einsum('a,b,c->abc', cols[0][c], cols[1][c], cols[2][c])
    return r


def _solve_col(res, u, v):
    """Solve for the W vector given U=u, V=v so that u⊗v⊗w matches `res`.
    Returns w (float array) if consistent, else None."""
    m = len(v)
    n = len(u)
    p = len(res)
    A = np.zeros((p * n, m))
    for j in range(m):
        for i in range(n):
            A[i * n + j, j] = u[i]
    b = np.zeros(p * n)
    for k in range(p):
        for i in range(n):
            b[i * n + k] = res[k, i]
    try:
        sol, resd, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None
    if rank < m:
        return None
    if np.linalg.norm(A @ sol - b) > 1e-8:
        return None
    w = sol
    # verify
    if np.linalg.norm(np.einsum('i,j,k->ijk', u, v, w) - res) > 1e-7:
        return None
    return w


def _build_target(n, m, p):
    T = np.zeros((p, n, m))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k, i, j] = 1.0
    return T


def _greedy_peel(T, alphabet, n, m, p, max_cols, rng):
    """Greedy: repeatedly pick a (u,v) that solves the residual, peel it.
    Returns list of (u,v,w) or None if stuck before covering T."""
    res = T.copy()
    cols = []
    for _ in range(max_cols):
        if not _time_ok():
            return None
        found = False
        # sample (u,v) pairs; include all-zero-free ones
        for u in alphabet:
            for v in alphabet:
                w = _solve_col(res, u, v, )
                if w is None:
                    continue
                # check w within a reasonable magnitude
                if np.max(np.abs(w)) > 6.0:
                    continue
                cols.append((u, v, w))
                res = res - np.einsum('i,j,k->ijk', u, v, w)
                found = True
                break
            if found:
                break
        if not found:
            # try to finish with schoolbook on remaining
            break
        if np.linalg.norm(res) < 1e-9:
            break
    # finish remainder with schoolbook
    if np.linalg.norm(res) > 1e-9:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(res[k, i, j]) > 1e-9:
                        u = np.zeros(m); u[j] = 1.0
                        v = np.zeros(p); v[k] = 1.0
                        w = np.zeros(n); w[i] = res[k, i, j]
                        cols.append((u, v, w))
                        res = res - np.einsum('i,j,k->ijk', u, v, w)
    if np.linalg.norm(res) > 1e-7:
        return None
    return cols


def _make_alphabet(rng, n, m, p):
    """Generate a list of candidate (u,v) unit-ish vectors over small integers."""
    base = [1, -1, 2, -2]
    us, vs = [], []
    # standard basis
    for j in range(m):
        u = np.zeros(m); u[j] = 1.0; us.append(u)
    for k in range(p):
        v = np.zeros(p); v[k] = 1.0; vs.append(v)
    # random small-integer vectors
    for _ in range(40):
        u = np.array([rng.choice(base) for _ in range(m)], dtype=float)
        us.append(u)
    for _ in range(60):
        v = np.array([rng.choice(base) for _ in range(p)], dtype=float)
        vs.append(v)
    return us, vs


def _verify(U, V, W, n, m, p):
    """Check that the columns reproduce the multiplication tensor exactly."""
    r = len(U[0])
    T = np.zeros((p, n, m))
    for c in range(r):
        u = np.array(U[c])
        v = np.array(V[c])
        w = np.array(W[c])
        T += np.einsum('i,j,k->ijk', u, v, w)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[k, i, j] - 1.0) > 1e-6:
                    return False
    return True


def _to_result(cols, n, m, p):
    rank = len(cols)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    for c, (u, v, w) in enumerate(cols):
        for j in range(m):
            U[j, c] = float(u[j])
        for k in range(p):
            V[j * p + k, c] = float(v[k])
        for i in range(n):
            W[k * n + i, c] = float(w[i])
    return {"U": U, "V": V, "W": W}


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    rng = random.Random(12345)
    T = _build_target(n, m, p)
    best = _schoolbook(n, m, p)

    for trial in range(200):
        if not _time_ok():
            break
        us, vs = _make_alphabet(rng, n, m, p)
        alphabet = list(itertools.product(us, vs))
        rng.shuffle(alphabet)
        cols = _greedy_peel(T, alphabet, n, m, p, 14, rng)
        if cols is None:
            continue
        if len(cols) >= len(best["U"][0]):
            continue
        res = _to_result(cols, n, m, p)
        if _verify(res["U"], res["V"], res["W"], n, m, p):
            best = res
            if len(cols) <= 14:
                break
    return best
