"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-7):
    """Verify that U, V, W give a valid tensor decomposition of matrix multiplication."""
    if not U or not V or not W:
        return False
    r = len(U[0]) if U else 0
    if r == 0:
        return False
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    if Ua.shape != (n*m, r) or Va.shape != (m*p, r) or Wa.shape != (p*n, r):
        return False
    T = np.zeros((n, m, p))
    for c in range(r):
        uc = Ua[:, c].reshape(n, m)
        vc = Va[:, c].reshape(m, p)
        wc = Wa[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                if uc[i, j] != 0:
                    for k in range(p):
                        T[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _residuals(params, n, m, p, r):
    """Residual vector of the tensor equation for given parameters."""
    U = params[:n*m*r].reshape(n*m, r)
    V = params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
    W = params[n*m*r + m*p*r :].reshape(p*n, r)
    res = np.empty(n*m*p)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                res[idx] = np.dot(U[i*m+j], V[j*p+k]*W[k*n+i]) - 1.0
                idx += 1
    return res

def _optimize_rank(n, m, p, r, deadline, n_restarts=8, seed_base=0):
    """
    Try to find a rank-r decomposition via nonlinear least squares.
    Returns (U, V, W) as lists or None.
    """
    from scipy.optimize import least_squares
    rng = np.random.RandomState(seed_base + r)
    best_cost = np.inf
    best_params = None

    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        # Random initialization
        U0 = rng.randn(n*m, r)
        V0 = rng.randn(m*p, r)
        W0 = rng.randn(p*n, r)
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        try:
            res = least_squares(_residuals, x0, args=(n, m, p, r),
                                method='lm', max_nfev=2000)
            if res.cost < best_cost:
                best_cost = res.cost
                best_params = res.x
            if best_cost < 1e-10:
                break
        except Exception:
            continue

    if best_params is None or best_cost > 1e-8:
        return None

    U = best_params[:n*m*r].reshape(n*m, r)
    V = best_params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
    W = best_params[n*m*r + m*p*r :].reshape(p*n, r)
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication.
    Uses schoolbook as baseline and searches for lower ranks via nonlinear optimization.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # 5-second safety margin before 120s

    # Schoolbook baseline
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p

    if not _verify(U_sb, V_sb, W_sb, n, m, p):
        return {"U": U_sb, "V": V_sb, "W": W_sb}

    # Lower bound on rank: max(n*m, m*p, p*n)
    lb = max(n*m, m*p, p*n)  # max(12, 20, 15) = 20

    # Search ranks from a reasonable lower bound up to schoolbook - 1
    # We cap the search to keep within time budget; try lower ranks first
    # since lower rank is the goal.
    search_start = max(lb, 24)
    search_end = best_rank - 1

    # Allocate time: give each rank attempt a slice of the budget
    n_ranks = max(1, search_end - search_start + 1)
    per_rank_time = (deadline - start_time) / n_ranks

    for target_rank in range(search_start, search_end + 1):
        if time.time() > deadline - 3:
            break
        rank_deadline = min(deadline - 3, time.time() + max(2.0, per_rank_time))
        result = _optimize_rank(n, m, p, target_rank, rank_deadline,
                                n_restarts=6, seed_base=target_rank * 101)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p, tol=1e-6):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                # Found a valid lower-rank decomposition; stop (it's the best
                # we can get at this rank, and we searched in increasing order)
                break

    # Final verification; fall back to schoolbook if anything is off
    if not _verify(best_U, best_V, best_W, n, m, p, tol=1e-6):
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = n * m * p

    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
