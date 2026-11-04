"""Rank-minimal exact tensor decomposition for 3x4x5 matrix multiplication.

The multiplication tensor T has entries (indexed by (i,a) in [0,n*m), (b,k) in [0,m*p), (k,i) in [0,p*n))
    T[i*m+a, b*p+k, k*n+i] = delta_{a,b}
A rank-R decomposition is U (n*m x R), V (m*p x R), W (p*n x R) with
    sum_c U[i*m+a,c] * V[b*p+k,c] * W[k*n+i,c] = delta_{a,b}.
We find a valid decomposition via ALS (linear-in-each-factor least squares)
and then attempt rank reduction. Lower rank = better score.
"""
import time
import numpy as np

n, m, p = 3, 4, 5
N1, N2, N3 = n * m, m * p, p * n  # 12, 20, 15
RANK_SCHOOLBOOK = n * m * p  # 60

_DEADLINE = time.time() + 140.0

def _remaining():
    return _DEADLINE - time.time()

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def _verify(U, V, W, tol=1e-8):
    """Return max absolute error of the bilinear reconstruction."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    R = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for k in range(p):
            # A_ik[a,b] = sum_c U[i*m+a,c]*W[k*n+i,c]*V[b*p+k,c]
            D_ik = U[i * m:(i + 1) * m, :] * W[k * n + i, :]      # m x R
            V_k = np.array([V[b * p + k, :] for b in range(m)])    # m x R
            A_ik = D_ik @ V_k.T                                     # m x m
            err = np.max(np.abs(A_ik - np.eye(m)))
            if err > max_err:
                max_err = err
    return max_err

# ---------------------------------------------------------------------------
# Schoolbook rank-60 decomposition (exact, all-ones pattern)
# ---------------------------------------------------------------------------
def _schoolbook():
    R = RANK_SCHOOLBOOK
    U = np.zeros((N1, R)); V = np.zeros((N2, R)); W = np.zeros((N3, R))
    col = 0
    for i in range(n):
        for a in range(m):
            for k in range(p):
                U[i * m + a, col] = 1.0
                V[a * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

# ---------------------------------------------------------------------------
# ALS: linear-in-each-factor least-squares updates.
# The tensor is linear in each factor when the other two are fixed, so each
# row of a factor is an independent small linear least-squares problem.
# ---------------------------------------------------------------------------
def _als(U, V, W, max_iters=200, tol=1e-12, verbose=False):
    U = np.array(U, dtype=float); V = np.array(V, dtype=float); W = np.array(W, dtype=float)
    R = U.shape[1]
    prev_err = _verify(U, V, W)
    for it in range(max_iters):
        if _remaining() < 8.0:
            break
        # ---- update U: for each row (i,a), solve LS over (k,b) ----
        for i in range(n):
            for a in range(m):
                # Build matrix A: (p*m) x R,  rhs: (p*m)
                rows = []
                rhs = []
                for k in range(p):
                    for b in range(m):
                        rows.append(V[b * p + k, :] * W[k * n + i, :])
                        rhs.append(1.0 if a == b else 0.0)
                A = np.array(rows)
                y = np.array(rhs)
                sol, *_ = np.linalg.lstsq(A, y, rcond=None)
                U[i * m + a, :] = sol
        # ---- update V: for each row (b,k), solve LS over (i,a) ----
        for b in range(m):
            for k in range(p):
                rows = []
                rhs = []
                for i in range(n):
                    for a in range(m):
                        rows.append(U[i * m + a, :] * W[k * n + i, :])
                        rhs.append(1.0 if a == b else 0.0)
                A = np.array(rows)
                y = np.array(rhs)
                sol, *_ = np.linalg.lstsq(A, y, rcond=None)
                V[b * p + k, :] = sol
        # ---- update W: for each row (k,i), solve LS over (i,a) ----
        for k in range(p):
            for i in range(n):
                rows = []
                rhs = []
                for a in range(m):
                    for b in range(m):
                        rows.append(U[i * m + a, :] * V[b * p + k, :])
                        rhs.append(1.0 if a == b else 0.0)
                A = np.array(rows)
                y = np.array(rhs)
                sol, *_ = np.linalg.lstsq(A, y, rcond=None)
                W[k * n + i, :] = sol

        err = _verify(U, V, W)
        if verbose:
            print(f"  ALS it={it} err={err:.3e}")
        if err < tol or abs(prev_err - err) < 1e-15:
            break
        prev_err = err
    return U, V, W, err

# ---------------------------------------------------------------------------
# Rank reduction: given a valid rank-R decomposition, try to drop a column
# (set it to zero) if the residual stays within tolerance. This is a greedy
# "snip" that can lower the effective rank.
# ---------------------------------------------------------------------------
def _snip(U, V, W, tol=1e-7, max_rounds=50):
    """Greedy column removal. Returns (U, V, W, effective_rank)."""
    U = np.array(U, dtype=float); V = np.array(V, dtype=float); W = np.array(W, dtype=float)
    R = U.shape[1]
    active = list(range(R))
    for _ in range(max_rounds):
        if _remaining() < 6.0:
            break
        if not active:
            break
        best_col = None
        best_err = None
        for c in active:
            U2 = U.copy(); V2 = V.copy(); W2 = W.copy()
            U2[:, c] = 0.0; V2[:, c] = 0.0; W2[:, c] = 0.0
            err = _verify(U2, V2, W2)
            if err <= tol:
                best_col = c
                best_err = err
                break  # take first removable
        if best_col is None:
            break
        U[:, best_col] = 0.0
        V[:, best_col] = 0.0
        W[:, best_col] = 0.0
        active.remove(best_col)
    eff_rank = sum(1 for c in range(R) if np.any(U[:, c] != 0) or np.any(V[:, c] != 0) or np.any(W[:, c] != 0))
    return U, V, W, eff_rank

# ---------------------------------------------------------------------------
# Random restart + ALS to find a valid decomposition at a target rank.
# ---------------------------------------------------------------------------
def _find_rank(R, n_restarts=8, seed=0):
    """Try to find a valid rank-R decomposition via random init + ALS."""
    if _remaining() < 12.0:
        return None
    best = None
    best_err = float('inf')
    for r in range(n_restarts):
        if _remaining() < 8.0:
            break
        rng = np.random.RandomState(seed + r * 7919)
        U = rng.randn(N1, R)
        V = rng.randn(N2, R)
        W = rng.randn(N3, R)
        # normalize columns to help conditioning
        for M in (U, V, W):
            nn = np.linalg.norm(M, axis=0, keepdims=True) + 1e-12
            M /= nn
        U, V, W, err = _als(U, V, W, max_iters=120)
        if err < best_err:
            best_err = err
            best = (U.copy(), V.copy(), W.copy(), err)
        if err < 1e-9:
            break
    if best is not None and best[3] < 1e-7:
        return best[:3]
    return None

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 140.0

    # Fallback: exact schoolbook rank 60.
    U_sb, V_sb, W_sb = _schoolbook()
    best = (U_sb, V_sb, W_sb, RANK_SCHOOLBOOK)

    # 1) Try to polish the schoolbook with ALS (may not reduce rank but ensures validity).
    if _remaining() > 30.0:
        U, V, W, err = _als(U_sb, V_sb, W_sb, max_iters=40)
        if err < 1e-8:
            U, V, W, eff = _snip(U, V, W)
            if eff < best[3]:
                best = (U, V, W, eff)

    # 2) Search for lower ranks via random-restart ALS, from the theoretical
    #    lower bound (20) upward. Stop at the first rank that yields a valid
    #    decomposition (greedy: we want the minimum, so go low-to-high).
    lower = max(N1, N2, N3)  # 20
    for R in range(lower, best[3]):
        if _remaining() < 25.0:
            break
        res = _find_rank(R, n_restarts=6, seed=R)
        if res is not None:
            U, V, W = res
            # polish + snip
            U, V, W, err = _als(U, V, W, max_iters=60)
            U, V, W, eff = _snip(U, V, W)
            if _verify(U, V, W) < 1e-7:
                best = (U, V, W, eff)
                break  # found the lowest feasible R; stop.

    U, V, W, rank = best
    # Final high-precision polish on the best found.
    if _remaining() > 15.0:
        U, V, W, err = _als(U, V, W, max_iters=80, tol=1e-13)
        if _verify(U, V, W) < 1e-8:
            U, V, W, eff = _snip(U, V, W)
            if _verify(U, V, W) < 1e-7:
                best = (U, V, W, eff)

    U, V, W, _ = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
