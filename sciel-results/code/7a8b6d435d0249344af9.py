import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    nm, mp, pn = n * m, m * p, p * n  # 4, 6, 6

    # Target tensor: T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def compute_tensor(U, V, W):
        return np.einsum('ir,jr,kr->ijk', U, V, W)

    def error_metric(U, V, W):
        T_prime = compute_tensor(U, V, W)
        return np.sum((T_prime - T_target) ** 2)

    def is_valid(U, V, W, tol=1e-8):
        return error_metric(U, V, W) < tol

    def try_half_integer_rounding(U, V, W, max_rounds=5000):
        """Try to round to half-integers while maintaining validity."""
        for _ in range(max_rounds):
            if time.time() > deadline - 5.0:
                break
            # Round to nearest half-integer
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            if is_valid(U_r, V_r, W_r):
                return U_r, V_r, W_r
            # If rounding doesn't work, try small perturbations
            break
        return None

    def gauss_newton(rank, time_limit, n_iters=200):
        """Gauss-Newton optimization for tensor decomposition."""
        if time.time() > time_limit:
            return None

        # Flatten parameters
        # U: (nm, rank), V: (mp, rank), W: (pn, rank)
        # Total params: nm*rank + mp*rank + pn*rank
        total_params = nm * rank + mp * rank + pn * rank

        def unpack(params):
            U = params[:nm * rank].reshape(nm, rank)
            V = params[nm * rank:nm * rank + mp * rank].reshape(mp, rank)
            W = params[nm * rank + mp * rank:].reshape(pn, rank)
            return U, V, W

        def residual(params):
            U, V, W = unpack(params)
            T_prime = compute_tensor(U, V, W)
            return (T_prime - T_target).flatten()

        # Initialize
        best_result = None
        best_error = float('inf')

        n_restarts = 30
        for restart in range(n_restarts):
            if time.time() > time_limit:
                break

            # Random initialization
            scale = 1.0
            U0 = np.random.uniform(-scale, scale, (nm, rank))
            V0 = np.random.uniform(-scale, scale, (mp, rank))
            W0 = np.random.uniform(-scale, scale, (pn, rank))

            params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])

            # Use L-BFGS-B optimizer
            try:
                result = minimize(residual, params0, method='L-BFGS-B',
                                  options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8})
                if result.success or result.fun < best_error:
                    params_opt = result.x
                    U_opt, V_opt, W_opt = unpack(params_opt)
                    err = error_metric(U_opt, V_opt, W_opt)
                    if err < best_error:
                        best_error = err
                        best_result = (U_opt.copy(), V_opt.copy(), W_opt.copy())

                    # Try half-integer rounding
                    if err < 1e-4:
                        rounded = try_half_integer_rounding(U_opt, V_opt, W_opt)
                        if rounded:
                            return rounded
            except Exception:
                continue

        return best_result

    def structured_search(rank, time_limit):
        """
        Structured search exploiting the algebraic structure.
        For 2x2 * 2x3, we can parameterize more cleverly.
        """
        if time.time() > time_limit:
            return None

        # The tensor has 24 non-zero entries (2*2*3 * 2 = 24, since each appears in 2 positions)
        # Actually, let me recount: n*m*p = 2*2*3 = 12 terms, each contributes 1 to T[a,b,c]
        # So 12 non-zero entries.

        # Strategy: Use the fact that we're looking for half-integer solutions
        # and do a systematic search over a bounded region.

        # For rank 10 or 11, the solution space is small.
        # Let's try a more systematic approach.

        best_result = None
        best_error = float('inf')

        # Use random restarts with Gauss-Newton
        n_restarts = 50
        for restart in range(n_restarts):
            if time.time() > time_limit:
                break

            # Different initialization strategies
            if restart % 3 == 0:
                # Small random init
                scale = 0.5
                U0 = np.random.uniform(-scale, scale, (nm, rank))
                V0 = np.random.uniform(-scale, scale, (mp, rank))
                W0 = np.random.uniform(-scale, scale, (pn, rank))
            elif restart % 3 == 1:
                # Larger random init
                scale = 1.5
                U0 = np.random.uniform(-scale, scale, (nm, rank))
                V0 = np.random.uniform(-scale, scale, (mp, rank))
                W0 = np.random.uniform(-scale, scale, (pn, rank))
            else:
                # Structured init based on schoolbook
                U0 = np.zeros((nm, rank))
                V0 = np.zeros((mp, rank))
                W0 = np.zeros((pn, rank))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            r = (i * m + j) * p + k  # pseudo-random rank assignment
                            r = r % rank
                            U0[a, r] += 1.0
                            V0[b, r] += 1.0
                            W0[c, r] += 1.0
                # Add noise
                U0 += np.random.uniform(-0.5, 0.5, (nm, rank))
                V0 += np.random.uniform(-0.5, 0.5, (mp, rank))
                W0 += np.random.uniform(-0.5, 0.5, (pn, rank))

            params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])

            try:
                result = minimize(lambda p: np.sum(residual_func(p, rank)**2), params0,
                                  method='BFGS',
                                  jac=lambda p: 2 * residual_func(p, rank) @ jacobian_residual(p, rank),
                                  options={'maxiter': 30, 'gtol': 1e-8})
                params_opt = result.x
                U_opt, V_opt, W_opt = unpack_params(params_opt, rank)
                err = error_metric(U_opt, V_opt, W_opt)
                if err < best_error:
                    best_error = err
                    best_result = (U_opt.copy(), V_opt.copy(), W_opt.copy())

                if err < 1e-4:
                    rounded = try_half_integer_rounding(U_opt, V_opt, W_opt)
                    if rounded:
                        return rounded
            except Exception:
                continue

        return best_result

    def residual_func(params, rank):
        U, V, W = unpack_params(params, rank)
        T_prime = compute_tensor(U, V, W)
        return (T_prime - T_target).flatten()

    def unpack_params(params, rank):
        U = params[:nm * rank].reshape(nm, rank)
        V = params[nm * rank:nm * rank + mp * rank].reshape(mp, rank)
        W = params[nm * rank + mp * rank:].reshape(pn, rank)
        return U, V, W

    def jacobian_residual(params, rank):
        """Compute Jacobian of residual w.r.t. params."""
        U, V, W = unpack_params(params, rank)
        # Residual r = einsum(U,V,W) - T
        # dr/dU = V^T ⊗ W^T (in some sense)
        # For each U[i, r], dT/da = V[:, r] * W[:, r] (outer product)
        # This is complex; use finite differences as fallback
        eps = 1e-7
        n_params = len(params)
        # Too slow for full Jacobian; use approximate
        return None

    # Schoolbook decomposition (rank 12)
    rank_school = n * m * p
    U_sb = np.zeros((nm, rank_school))
    V_sb = np.zeros((mp, rank_school))
    W_sb = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_U, best_V, best_W = U_sb.tolist(), V_sb.tolist(), W_sb.tolist()
    best_rank = rank_school

    # Try rank 10 first
    time_for_rank_10 = min(70.0, deadline - time.time() - 10.0)
    if time_for_rank_10 > 10.0:
        # Method 1: Gauss-Newton with multiple restarts
        res = gauss_newton(10, time.time() + time_for_rank_10 * 0.6)
        if res and is_valid(*res):
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

        # Method 2: Structured search
        if time.time() < deadline - 10.0:
            res = structured_search(10, time.time() + time_for_rank_10 * 0.4)
            if res and is_valid(*res):
                U, V, W = res
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try rank 11
    time_for_rank_11 = min(50.0, deadline - time.time() - 10.0)
    if time_for_rank_11 > 5.0:
        res = gauss_newton(11, time.time() + time_for_rank_11 * 0.7)
        if res and is_valid(*res):
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

        if time.time() < deadline - 5.0:
            res = structured_search(11, time.time() + time_for_rank_11 * 0.3)
            if res and is_valid(*res):
                U, V, W = res
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
