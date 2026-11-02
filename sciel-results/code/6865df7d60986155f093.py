"""Program for matrix multiplication 3x4 * 4x6 with rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]
n, m, p = DIMENSIONS


def _build_target():
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _verify(U, V, W, T, tol=1e-6):
    try:
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
    except Exception:
        return False
    if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    r = U.shape[1]
    if V.shape[1] != r or W.shape[1] != r:
        return False
    if not (np.all(np.isfinite(U)) and np.all(np.isfinite(V)) and np.all(np.isfinite(W))):
        return False
    Tcalc = np.einsum('ar,br,cr->abc', U, V, W)
    return bool(np.max(np.abs(Tcalc - T)) <= tol)


def _schoolbook():
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _als(r, seed, T, max_sweeps, start_time, deadline):
    rng = np.random.default_rng(seed)
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    for col in range(r):
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, m))
        k = int(rng.integers(0, p))
        U[i * m + j, col] = 1.0
        V[j * p + k, col] = 1.0
        W[k * n + i, col] = 1.0
    U += rng.standard_normal((n * m, r)) * 0.02
    V += rng.standard_normal((m * p, r)) * 0.02
    W += rng.standard_normal((p * n, r)) * 0.02

    aa, bb, cc = np.meshgrid(np.arange(n * m), np.arange(m * p), np.arange(p * n), indexing='ij')
    A_idx = aa.ravel(); B_idx = bb.ravel(); C_idx = cc.ravel()
    y_all = T.ravel()

    for _ in range(max_sweeps):
        if time.time() - start_time > deadline:
            break
        for a in range(n * m):
            mask = A_idx == a
            X = V[B_idx[mask], :] * W[C_idx[mask], :]
            U[a, :] = np.linalg.lstsq(X, y_all[mask], rcond=None)[0]
        for b in range(m * p):
            mask = B_idx == b
            X = U[A_idx[mask], :] * W[C_idx[mask], :]
            V[b, :] = np.linalg.lstsq(X, y_all[mask], rcond=None)[0]
        for c in range(p * n):
            mask = C_idx == c
            X = U[A_idx[mask], :] * V[B_idx[mask], :]
            W[c, :] = np.linalg.lstsq(X, y_all[mask], rcond=None)[0]
    return U, V, W


def _round_half(U, V, W):
    return (np.round(U * 2) / 2, np.round(V * 2) / 2, np.round(W * 2) / 2)


def construct():
    start_time = time.time()
    DEADLINE = 110.0

    def check_time():
        return time.time() - start_time < DEADLINE

    T = _build_target()

    # Baseline: schoolbook (guaranteed correct, rank 72)
    best = _schoolbook()
    best_rank = n * m * p
    assert _verify(best["U"], best["V"], best["W"], T)

    # Search for lower rank via ALS + rounding + exact verification.
    ranks_to_try = list(range(best_rank - 1, 39, -1))
    seeds_per_rank = 2
    for r in ranks_to_try:
        if not check_time():
            break
        for seed in range(seeds_per_rank):
            if not check_time():
                break
            # Allocate sweeps based on remaining time; cap for speed
            remaining = DEADLINE - (time.time() - start_time)
            max_sweeps = max(30, min(300, int(remaining * 2)))
            try:
                U, V, W = _als(r, seed, T, max_sweeps, start_time, DEADLINE)
            except Exception:
                continue
            # Round to half-integers and verify
            Ur, Vr, Wr = _round_half(U, V, W)
            if _verify(Ur.tolist(), Vr.tolist(), Wr.tolist(), T):
                best = {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}
                best_rank = r
                break  # move to next (lower) r
    return best
