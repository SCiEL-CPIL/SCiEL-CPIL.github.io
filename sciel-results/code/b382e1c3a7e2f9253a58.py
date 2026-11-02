"""Tensor decomposition search for 3x5x5 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p

    # Build schoolbook as guaranteed fallback
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = schoolbook_rank

    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def residual(U, V, W):
        """Compute sum of squared differences from the target tensor."""
        Ur = U.reshape(n, m, -1)
        Vr = V.reshape(m, p, -1)
        Wr = W.reshape(p, n, -1)
        T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
        return float(np.sum((T - 1.0) ** 2))

    def try_rank(rank, n_attempts, max_iter, structured=False):
        """Attempt to find a rank-r decomposition. Returns (U, V, W) or None."""
        for attempt in range(n_attempts):
            if time.time() > deadline - 15:
                return None

            if structured and attempt % 2 == 0:
                # Structured initialization: sparse random with small integers
                U = np.random.randint(0, 3, (n * m, rank)) - 1
                V = np.random.randint(0, 3, (m * p, rank)) - 1
                W = np.random.randint(0, 3, (p * n, rank)) - 1
                # Sparsify: zero out ~50% of entries
                mask_u = np.random.rand(n * m, rank) > 0.5
                mask_v = np.random.rand(m * p, rank) > 0.5
                mask_w = np.random.rand(p * n, rank) > 0.5
                U *= mask_u
                V *= mask_v
                W *= mask_w
            else:
                # Random initialization with structured scaling
                U = np.random.randn(n * m, rank) * np.sqrt(1.0 / rank)
                V = np.random.randn(m * p, rank) * np.sqrt(1.0 / rank)
                W = np.random.randn(p * n, rank) * np.sqrt(1.0 / rank)

            def objective(params):
                U_f = params[:n * m * rank].reshape(n * m, rank)
                V_f = params[n * m * rank:n * m * rank + m * p * rank].reshape(m * p, rank)
                W_f = params[n * m * rank + m * p * rank:].reshape(p * n, rank)
                return residual(U_f, V_f, W_f)

            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

            try:
                res = minimize(objective, params0, method='L-BFGS-B',
                               options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
            except Exception:
                continue

            if res.fun < 1e-10:
                # Polish with tighter settings
                try:
                    res2 = minimize(objective, res.x, method='L-BFGS-B',
                                    options={'maxiter': max_iter * 2, 'ftol': 1e-16, 'gtol': 1e-12})
                    if res2.fun < res.fun:
                        res = res2
                except Exception:
                    pass

            if res.fun < 1e-10:
                U_opt = res.x[:n * m * rank].reshape(n * m, rank)
                V_opt = res.x[n * m * rank:n * m * rank + m * p * rank].reshape(m * p, rank)
                W_opt = res.x[n * m * rank + m * p * rank:].reshape(p * n, rank)

                # Snap to nearest integers (tensor entries are integers)
                U_snap = np.round(U_opt)
                V_snap = np.round(V_opt)
                W_snap = np.round(W_opt)

                r_snapped = residual(U_snap, V_snap, W_snap)
                if r_snapped < 1e-9:
                    return (U_snap, V_snap, W_snap)

                # Also try snapping to half-integers
                U_snap2 = np.round(U_opt * 2.0) / 2.0
                V_snap2 = np.round(V_opt * 2.0) / 2.0
                W_snap2 = np.round(W_opt * 2.0) / 2.0
                r_snapped2 = residual(U_snap2, V_snap2, W_snap2)
                if r_snapped2 < 1e-9:
                    return (U_snap2, V_snap2, W_snap2)

        return None

    # Known bounds: for 3x5x5, schoolbook is 75.
    # Lower bounds from literature suggest rank >= 18 or so for small matrices.
    # Try from low to high, stopping at first success.
    min_rank = 18
    max_rank = min(50, schoolbook_rank - 1)

    for rank in range(min_rank, max_rank + 1):
        if time.time() > deadline - 20:
            break

        remaining = deadline - 15 - time.time()
        if remaining <= 0:
            break

        # Budget per rank: more attempts for lower ranks
        attempts = max(3, int(remaining * 0.05))
        attempts = min(attempts, 30)
        max_iter = 300

        result = try_rank(rank, attempts, max_iter, structured=True)
        if result is not None:
            U_r, V_r, W_r = result
            best = {
                "U": U_r.tolist(),
                "V": V_r.tolist(),
                "W": W_r.tolist()
            }
            best_rank = rank
            break

    return best
