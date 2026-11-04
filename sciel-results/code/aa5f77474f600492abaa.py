"""Matrix multiplication 2x3x3 tensor decomposition - rank minimal."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 3]

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
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > 1e-7:
                    return False
    return True


def _build_residual_function(n, m, p, rank):
    """Build a vectorized residual function for rank-r decomposition."""
    nm, mp, pn = n * m, m * p, p * n

    def _residuals(params):
        u_size = nm * rank
        v_size = mp * rank
        U = params[:u_size].reshape(nm, rank)
        V = params[u_size:u_size + v_size].reshape(mp, rank)
        W = params[u_size + v_size:].reshape(pn, rank)

        # Compute tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        T = np.zeros((n, m, p))
        for r in range(rank):
            Ur = U[:, r]  # shape (nm,)
            Vr = V[:, r]  # shape (mp,)
            Wr = W[:, r]  # shape (pn,)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += Ur[i * m + j] * Vr[j * p + k] * Wr[k * n + i]
        return (T - 1.0).flatten()

    return _residuals


def _try_structured_decomposition(rank):
    """
    Try to find a rank-rank decomposition using structured initialization
    based on the algebraic structure of matrix multiplication.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    from scipy.optimize import least_squares

    _residuals = _build_residual_function(n, m, p, rank)

    best_result = None
    best_cost = float('inf')

    # Strategy 1: Start from a near-schoolbook initialization and prune
    # Strategy 2: Random with different scales
    # Strategy 3: Structured based on output positions

    n_attempts = 12
    for attempt in range(n_attempts):
        if _remaining() < 3.0:
            break

        if attempt == 0:
            # Near-schoolbook: start with rank-18 and zero out some columns
            U = np.zeros((nm, rank))
            V = np.zeros((mp, rank))
            W = np.zeros((pn, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col < rank:
                            U[i * m + j, col] = 1.0
                            V[j * p + k, col] = 1.0
                            W[k * n + i, col] = 1.0
                        col += 1
            # Add small random perturbation to break symmetry
            U += np.random.randn(nm, rank) * 0.05
            V += np.random.randn(mp, rank) * 0.05
            W += np.random.randn(pn, rank) * 0.05
            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        elif attempt == 1:
            # Structured: group by output (i,k)
            U = np.zeros((nm, rank))
            V = np.zeros((mp, rank))
            W = np.zeros((pn, rank))
            col = 0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        if col < rank:
                            U[i * m + j, col] = 1.0
                            V[j * p + k, col] = 1.0
                            W[k * n + i, col] = 1.0
                        col += 1
            U += np.random.randn(nm, rank) * 0.1
            V += np.random.randn(mp, rank) * 0.1
            W += np.random.randn(pn, rank) * 0.1
            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        else:
            # Random with varying scales
            scale = 0.3 + 0.1 * (attempt % 5)
            U = np.random.randn(nm, rank) * scale
            V = np.random.randn(mp, rank) * scale
            W = np.random.randn(pn, rank) * scale
            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

        try:
            result = least_squares(_residuals, params0, method='lm', max_nfev=3000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if result.cost < best_cost:
                best_cost = result.cost
                best_result = result
                if result.cost < 1e-14:
                    break
        except Exception:
            pass

    if best_result is None or best_cost > 1e-10:
        return None

    u_size = nm * rank
    v_size = mp * rank
    U = best_result.x[:u_size].reshape(nm, rank)
    V = best_result.x[u_size:u_size + v_size].reshape(mp, rank)
    W = best_result.x[u_size + v_size:].reshape(pn, rank)

    # Try rounding to half-integers
    for factor in [2, 4, 8]:
        U_r = np.round(U * factor) / factor
        V_r = np.round(V * factor) / factor
        W_r = np.round(W * factor) / factor
        params_r = np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()])
        res_r = _residuals(params_r)
        if np.max(np.abs(res_r)) < 1e-6:
            U, V, W = U_r, V_r, W_r
            break

    # Final verification
    if _verify(U, V, W, rank):
        return U.tolist(), V.tolist(), W.tolist()

    # Try rounding to integers
    U_i = np.round(U)
    V_i = np.round(V)
    W_i = np.round(W)
    if _verify(U_i, V_i, W_i, rank):
        return U_i.tolist(), V_i.tolist(), W_i.tolist()

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


def _try_winograd_like():
    """
    Try a Winograd-style approach: exploit that for fixed (i,k),
    the inner product over j can be computed with fewer multiplications
    if we use clever combinations.
    
    For 2x3 * 3x3, output is 2x3. For each output (i,k), we compute:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We can try to find a low-rank decomposition by considering the
    structure more carefully.
    """
    n, m, p = DIMENSIONS
    # Try ranks from 12 to 17
    for rank in range(12, 18):
        if _remaining() < 5.0:
            break
        result = _try_structured_decomposition(rank)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    return None


def construct():
    """
    Find rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication.
    """
    global _DEADLINE
    _DEADLINE = time.time() + 140.0

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 18

    # First try to find a lower-rank decomposition
    result = _try_winograd_like()
    if result is not None:
        return result

    # Fallback: schoolbook decomposition
    return _schoolbook()
