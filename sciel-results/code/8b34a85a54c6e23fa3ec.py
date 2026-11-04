"""Matrix multiplication 2x3x3 tensor decomposition - rank minimal."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 140.0

def _remaining():
    return _DEADLINE - time.time()


def _verify(U, V, W, rank):
    """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    n, m, p = DIMENSIONS
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(val - 1.0) > 1e-7:
                    return False
    return True


def _try_rank_decomposition(rank, timeout=15.0):
    """
    Attempt to find a rank-`rank` decomposition of the all-ones tensor (2,3,3).
    Uses scipy least_squares with multiple random initializations.
    Returns (U, V, W) as lists if found, else None.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n

    def _residuals(params, rank):
        u_size = nm * rank
        v_size = mp * rank
        u_flat = params[:u_size]
        v_flat = params[u_size:u_size+v_size]
        w_flat = params[u_size+v_size:]

        U = u_flat.reshape(nm, rank)
        V = v_flat.reshape(mp, rank)
        W = w_flat.reshape(pn, rank)

        residuals = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    residuals[i, j, k] = val - 1.0
        return residuals.flatten()

    from scipy.optimize import least_squares

    best_result = None
    best_cost = float('inf')

    n_attempts = 8
    for attempt in range(n_attempts):
        if _remaining() < timeout:
            break
        np.random.seed(1000 + rank * 100 + attempt)
        scale = 0.5
        u0 = np.random.randn(nm, rank) * scale
        v0 = np.random.randn(mp, rank) * scale
        w0 = np.random.randn(pn, rank) * scale
        params0 = np.concatenate([u0.flatten(), v0.flatten(), w0.flatten()])

        try:
            result = least_squares(_residuals, params0, args=(rank,), method='lm', max_nfev=2000)
            if result.cost < best_cost:
                best_cost = result.cost
                best_result = result
                if result.cost < 1e-12:
                    break
        except Exception:
            pass

    if best_result is None:
        return None

    if best_cost > 1e-8:
        return None

    # Extract solution
    u_size = nm * rank
    v_size = mp * rank
    U = best_result.x[:u_size].reshape(nm, rank)
    V = best_result.x[u_size:u_size+v_size].reshape(mp, rank)
    W = best_result.x[u_size+v_size:].reshape(pn, rank)

    # Try to round to half-integers
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2

    # Verify rounded version
    params_r = np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()])
    res_r = _residuals(params_r, rank)
    if np.max(np.abs(res_r)) < 1e-6:
        U, V, W = U_r, V_r, W_r

    # Final verification
    if _verify(U, V, W, rank):
        return U.tolist(), V.tolist(), W.tolist()

    # Try rounding to integers
    U_i = np.round(U)
    V_i = np.round(V)
    W_i = np.round(W)
    params_i = np.concatenate([U_i.flatten(), V_i.flatten(), W_i.flatten()])
    res_i = _residuals(params_i, rank)
    if np.max(np.abs(res_i)) < 1e-6:
        U, V, W = U_i, V_i, W_i
        if _verify(U, V, W, rank):
            return U.tolist(), V.tolist(), W.tolist()

    return None


def _schoolbook():
    """Return the schoolbook rank-18 decomposition."""
    n, m, p = DIMENSIONS
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

    return {"U": U, "V": V, "W": W}


def construct():
    """
    Find rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication.
    Searches for lower-rank decompositions using optimization, falls back to schoolbook.
    """
    global _DEADLINE
    _DEADLINE = time.time() + 140.0

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 18

    # Theoretical lower bound: rank >= max(n*m, m*p, p*n) / something
    # For all-ones tensor, rank >= max(n, m, p) in general, but tighter bounds exist.
    # For (2,3,3), let's search from rank 10 upward.
    # Actually, the tensor rank of all-ones (n,m,p) is at least max(n,m,p).
    # For (2,3,3), lower bound is at least 3. But practical lower bounds are higher.
    # Let's try from rank 12 to 18 (18 is guaranteed to work).

    # Try ranks from low to high, stopping at first success
    for rank in range(12, schoolbook_rank + 1):
        if _remaining() < 5:
            break
        result = _try_rank_decomposition(rank, timeout=10.0)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook decomposition
    return _schoolbook()
# EVOLVE-BLOCK-END
