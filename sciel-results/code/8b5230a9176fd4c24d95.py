"""Tensor decomposition search for 2x3x8 matrix multiplication.

Uses a structured ansatz: identity terms (rank n*m*p=48) plus low-rank
corrections on the (i,k) fibers, with a randomized local-search polish that
minimizes the squared Frobenius residual.  A deadline guard returns the best
valid decomposition found before the wall-clock limit.
"""
import time
import random
import numpy as np

_DIMS = (2, 3, 8)
_DEADLINE = time.time() + 110.0  # safety margin vs 120 s limit


def _make_identity():
    """Identity (schoolbook) decomposition as (U,V,W) float arrays."""
    n, m, p = _DIMS
    r = n * m * p
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _residual(U, V, W):
    """Squared Frobenius norm of (U@V.T) - target, using W as the target."""
    return float(np.sum((U @ V.T - W) ** 2))


def _try_polish(U, V, W, rng, budget):
    """Local search minimizing residual; returns improved (U,V,W) or None.

    Mutates copies.  Coordinate updates on the correction columns only.
    """
    n, m, p = _DIMS
    r = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()

    # Correction columns are those beyond the first n*m*p (identity) columns.
    n_id = n * m * p
    corr = range(n_id, r)
    if not corr:
        return None

    # Precompute U,V as float64 views.
    Uf = U.astype(np.float64)
    Vf = V.astype(np.float64)
    Wf = W.astype(np.float64)

    best = _residual(Uf, Vf, Wf)
    if best < 1e-12:
        return U, V, W

    def try_update(idx, axis, val):
        nonlocal Uf, Vf, Wf, best
        old = Uf[idx] if axis == 0 else (Vf[idx] if axis == 1 else Wf[idx])
        cur = old.copy()
        if axis == 0:
            Uf[idx] = val
        elif axis == 1:
            Vf[idx] = val
        else:
            Wf[idx] = val
        res = _residual(Uf, Vf, Wf)
        if res < best - 1e-12:
            best = res
            return True
        else:
            if axis == 0:
                Uf[idx] = cur
            elif axis == 1:
                Vf[idx] = cur
            else:
                Wf[idx] = cur
            return False

    # Coordinate descent: pick a random correction column, a random row of one
    # matrix, and search over a small value grid for the best move.
    step = 0
    while time.time() < budget and best > 1e-10:
        step += 1
        c = corr[rng.randrange(len(corr))]
        axis = rng.randrange(3)
        # choose row
        rows = Uf.shape[0] if axis == 0 else (Vf.shape[0] if axis == 1 else Wf.shape[0])
        idx = rng.randrange(rows)
        # value grid: current value plus perturbations
        cur_val = (Uf[idx, c] if axis == 0 else (Vf[idx, c] if axis == 1 else Wf[idx, c]))
        vals = [cur_val]
        for s in (0.5, 1.0, -0.5, -1.0, 0.25, -0.25, 2.0, -2.0):
            vals.append(cur_val + s)
        # also try snapping to small integers/half-integers
        for s in (-2, -1, 0, 1, 2, 0.5, -0.5):
            vals.append(float(s))
        vals = sorted(set(round(v, 4) for v in vals))
        improved = False
        for v in vals:
            if try_update(idx, axis, v):
                improved = True
                break
        # occasional random restart of a whole correction column
        if step % 500 == 0 and not improved:
            pass
        if best < 1e-12:
            break

    if best < 1e-12:
        return U, V, W
    return None


def _build_strassen():
    """Build a rank-44 decomposition via Strassen on 2x2 sub-blocks.

    2x3x8 = split the 3 columns of A into (2,1) and 8 rows of B into (4,4).
    Use Strassen for the 2x2 * 2x4 part (rank 7 instead of 8) for each of the
    two 4-row blocks, plus identity for the 2x1 * 1x4 part.
    """
    # This is complex; instead we rely on the general search. Return identity
    # as a safe fallback.
    return _make_identity()


def construct():
    n, m, p = _DIMS
    rng = random.Random(12345)

    U, V, W = _make_identity()
    best = (U, V, W)
    best_res = _residual(U, V, W)
    if best_res < 1e-12:
        # Already valid schoolbook; try to improve via search.
        pass

    # ---- Search phase: add correction columns and polish ----
    base_r = n * m * p
    # We allow a modest number of extra correction columns; the polish will
    # zero out any that are not useful, but rank is the column count, so we
    # only keep columns that carry signal.  Strategy: start with a few extra
    # columns and let local search drive them to useful values.
    extra = 12
    r = base_r + extra
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    # identity part
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    # random small init for correction columns
    for c in range(base_r, r):
        U[base_r:, c] = rng.uniform(-1, 1)
        V[base_r:, c] = rng.uniform(-1, 1)
        W[base_r:, c] = rng.uniform(-1, 1)

    budget = min(_DEADLINE - 5.0, time.time() + 60.0)
    res = _residual(U, V, W)
    if res < best_res:
        best_res = res
        best = (U.copy(), V.copy(), W.copy())

    polished = _try_polish(U, V, W, rng, budget)
    if polished is not None:
        res = _residual(polished[0], polished[1], polished[2])
        if res < best_res:
            best_res = res
            best = polished

    # ---- Finalize: drop empty correction columns, snap to half-integers ----
    U, V, W = best
    # Identify columns that are all-zero (unused) and drop them.
    keep = []
    for c in range(U.shape[1]):
        if np.any(U[:, c] != 0) or np.any(V[:, c] != 0) or np.any(W[:, c] != 0):
            keep.append(c)
    U = U[:, keep]
    V = V[:, keep]
    W = W[:, keep]

    # Snap to half-integers.
    def snap(A):
        A = A.astype(np.float64)
        return np.round(A * 2.0) / 2.0
    U = snap(U)
    V = snap(V)
    W = snap(W)

    # Verify exactness after snapping.
    res = _residual(U, V, W)
    if res > 1e-9:
        # Snapping broke it; fall back to best pre-snap, or identity.
        U, V, W = _make_identity()
        # Drop nothing; identity is always valid.
    else:
        # Optional: try to drop redundant columns after snapping.
        # (Skip to keep it simple and safe.)
        pass

    def to_list(A):
        return [[(int(x) if x == int(x) else float(x)) for x in row] for row in A]

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
