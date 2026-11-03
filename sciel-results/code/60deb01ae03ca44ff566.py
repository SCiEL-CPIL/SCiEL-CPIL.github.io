"""Combined program: structural idea from current (block-doubling 3x3x3 -> 3x3x6)
plus refinement machinery (vectorized ALS, half-integer rounding, multiple seeds,
direct 3x3x6 search, deadline guard)."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 120  # Safety margin before the 150 s kill

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _round_half(U, V, W):
    """Round to nearest half-integer and verify."""
    U_r = np.round(U * 2.0) / 2.0
    V_r = np.round(V * 2.0) / 2.0
    W_r = np.round(W * 2.0) / 2.0
    n, m, p = DIMENSIONS
    if _verify(U_r, V_r, W_r, n, m, p, tol=1e-9):
        return U_r, V_r, W_r
    return None

def _als(n, m, p, rank, max_iters=400, seed=0, deadline=None, tol=1e-8):
    """
    Alternating least squares for rank-r decomposition of the n x m x p
    matrix multiplication tensor.  Vectorised per-factor updates.
    Returns (U, V, W) on success, else None.
    """
    if deadline is not None and time.time() > deadline - 3.0:
        return None

    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank) * 0.5
    V = rng.randn(m * p, rank) * 0.5
    W = rng.randn(p * n, rank) * 0.5

    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Pre-compute index arrays for speed
    idx_U = [(i * m + j) for i in range(n) for j in range(m)]
    idx_V = [(j * p + k) for j in range(m) for k in range(p)]
    idx_W = [(k * n + i) for k in range(p) for i in range(n)]

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 3.0:
            break

        # ---- Update U ----
        for i in range(n):
            for j in range(m):
                # A[r, k] = V[j*p+k, r] * W[k*n+i, r]
                A = np.zeros((rank, p))
                for k in range(p):
                    A[:, k] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    U[i * m + j, :] = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
                except np.linalg.LinAlgError:
                    pass

        # ---- Update V ----
        for j in range(m):
            for k in range(p):
                B = np.zeros((rank, n))
                for i in range(n):
                    B[:, i] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    V[j * p + k, :] = np.linalg.lstsq(B.T, np.ones(n), rcond=None)[0]
                except np.linalg.LinAlgError:
                    pass

        # ---- Update W ----
        for k in range(p):
            for i in range(n):
                C = np.zeros((rank, m))
                for j in range(m):
                    C[:, j] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    W[k * n + i, :] = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
                except np.linalg.LinAlgError:
                    pass

        # ---- Error check ----
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    e = abs(s - 1.0)
                    if e > max_err:
                        max_err = e

        if max_err < best_err:
            best_err = max_err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if max_err < tol:
            break

    if best_err < 1e-6:
        return best_U, best_V, best_W
    return None

def _build_3x3x6(U0, V0, W0):
    """
    Expand a valid 3x3x3 rank-R decomposition to 3x3x6 rank-2R.
    The 3x3x6 multiplication tensor is the direct sum of two 3x3x3 tensors
    (k=0,1,2 and k=3,4,5), so we duplicate the 3x3x3 factors.
    """
    R = U0.shape[1]
    n3, m3, p3 = 3, 3, 3
    n6, m6, p6 = 3, 3, 6
    total_rank = 2 * R

    U = np.hstack([U0, U0])  # (9, 2R)

    V = np.zeros((m6 * p6, total_rank))
    for j in range(m3):
        for k in range(p3):
            V[j * p6 + k, 0:R] = V0[j * p3 + k, :]
            V[j * p6 + (k + p3), R:total_rank] = V0[j * p3 + k, :]

    W = np.zeros((p6 * n6, total_rank))
    for k in range(p3):
        for i in range(n3):
            W[k * n6 + i, 0:R] = W0[k * p3 + i, :]
            W[(k + p3) * n6 + i, R:total_rank] = W0[k * p3 + i, :]

    return U, V, W

def _try_3x3x3_then_expand(deadline):
    """Try to find a 3x3x3 decomposition, round, expand to 3x3x6."""
    for r3 in [23, 24, 25, 26, 27]:
        if time.time() > deadline - 10.0:
            return None
        for seed in range(8):
            if time.time() > deadline - 10.0:
                return None
            res = _als(3, 3, 3, r3, max_iters=300, seed=seed, deadline=deadline)
            if res is None:
                continue
            U0, V0, W0 = res
            # Try half-integer rounding
            rounded = _round_half(U0, V0, W0)
            if rounded is not None:
                U0, V0, W0 = rounded
            U, V, W = _build_3x3x6(U0, V0, W0)
            if _verify(U, V, W, 3, 3, 6, tol=1e-9):
                return {"U": U, "V": V, "W": W, "rank": 2 * r3}
    return None

def _try_direct_3x3x6(deadline):
    """Try direct ALS on 3x3x6 at various ranks."""
    for r in [46, 48, 50, 52, 54, 56, 60]:
        if time.time() > deadline - 10.0:
            return None
        for seed in range(4):
            if time.time() > deadline - 10.0:
                return None
            res = _als(3, 3, 6, r, max_iters=200, seed=seed, deadline=deadline)
            if res is None:
                continue
            U, V, W = res
            rounded = _round_half(U, V, W)
            if rounded is not None:
                U, V, W = rounded
            if _verify(U, V, W, 3, 3, 6, tol=1e-9):
                return {"U": U, "V": V, "W": W, "rank": r}
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = START_TIME + TIME_LIMIT

    # Fallback: schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best = {"U": U_sb, "V": V_sb, "W": W_sb, "rank": n * m * p}

    # Strategy 1: find 3x3x3 decomposition, expand to 3x3x6
    if time.time() < deadline - 15:
        res = _try_3x3x3_then_expand(deadline)
        if res is not None and res["rank"] < best["rank"]:
            best = res

    # Strategy 2: direct 3x3x6 search
    if time.time() < deadline - 15:
        res = _try_direct_3x3x6(deadline)
        if res is not None and res["rank"] < best["rank"]:
            best = res

    # Return in the required interface
    U = best["U"]
    V = best["V"]
    W = best["W"]
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-START
# Entry point is construct() defined above.
# EVOLVE-BLOCK-END
