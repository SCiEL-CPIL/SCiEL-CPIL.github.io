"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.special import eval_genlaguerre, roots_laguerre
from scipy.optimize import minimize_scalar, differential_evolution, minimize

# EVOLVE-BLOCK-START
def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Returns a list of 12 positive root locations.
    Score is the upper bound on C4 (lower is better for the evaluator's raw value,
    but the search score is seed/value, so higher search score = better).
    """
    start_time = time.time()
    DEADLINE = 630.0  # seconds wall-clock limit
    SAFETY_MARGIN = 30.0  # return 30s before the kill

    def time_ok():
        return (time.time() - start_time) < (DEADLINE - SAFETY_MARGIN)

    m = 12
    n = 1

    # Precompute the transformation matrix from standard basis to Laguerre basis
    # P(x) = sum c_k L_k^{(1)}(x)
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    A = np.zeros((13, 13))  # A[k, j] = coeff of x^j in L_k^{(1)}(x)
    for k in range(13):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)

    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    def compute_coefficients(roots):
        """Compute Laguerre coefficients c_k for P(x) = prod(x - r_i)"""
        coeffs_desc = np.poly(roots)  # [1, -e1, e2, ..., (-1)^12 e12]
        p = coeffs_desc[::-1]          # ascending
        return A_T_inv @ p

    def compute_bound(roots):
        """
        Compute the upper bound on C4.
        The bound is related to 1 / ||P||^2_Laguerre.
        We want to minimize the bound, so we want to maximize ||P||^2.
        """
        if any(r <= 0 for r in roots):
            return 1e10
        c = compute_coefficients(roots)
        # Norm squared in Laguerre basis for n=1:
        # <L_k^{(1)}, L_l^{(1)}> = delta_{kl} * (k+1)
        norm_sq = sum(c[k]**2 * (k+1) for k in range(13))
        if norm_sq <= 1e-10:
            return 1e10
        return 1.0 / norm_sq

    def get_roots_from_alpha(alpha):
        """
        Get roots of L_12^{(alpha)}(x) using the companion matrix method.
        """
        # Coefficients of L_m^{(alpha)}(x) in descending order
        # Coeff of x^k is (-1)^k * binom(m+alpha, m-k) / k!
        coeffs = []
        for k in range(m, -1, -1):
            try:
                binom_val = math.gamma(m + alpha + 1) / (
                    math.gamma(m - k + 1) * math.gamma(k + alpha + 1))
            except (ValueError, OverflowError):
                return None
            coeff = ((-1)**k) * binom_val / math.factorial(k)
            coeffs.append(coeff)

        try:
            roots = np.roots(coeffs)
        except Exception:
            return None

        real_roots = []
        for r in roots:
            if abs(r.imag) < 1e-10:
                real_part = r.real
                if real_part > 1e-10:
                    real_roots.append(real_part)

        if len(real_roots) != m:
            return None

        return np.array(real_roots)

    # Strategy: 1D search over alpha (Laguerre-root family) + 12D log-space polish
    best_bound = 1e10
    best_roots = None
    best_alpha = None

    # Initial coarse scan
    alpha_min = -10.0
    alpha_max = 200.0
    num_points = 200

    for i in range(num_points):
        if not time_ok():
            break
        alpha = alpha_min + (alpha_max - alpha_min) * i / (num_points - 1)
        roots = get_roots_from_alpha(alpha)
        if roots is None:
            continue
        bound = compute_bound(roots)
        if bound < best_bound:
            best_bound = bound
            best_roots = roots.copy()
            best_alpha = alpha

    # Refine around the best alpha found
    if best_alpha is not None:
        def alpha_cost(alpha):
            roots = get_roots_from_alpha(alpha)
            if roots is None:
                return 1e10
            return compute_bound(roots)

        local_min = max(-10.0, best_alpha - 10.0)
        local_max = min(200.0, best_alpha + 10.0)

        try:
            res = minimize_scalar(alpha_cost, bounds=(local_min, local_max),
                                  method='bounded', options={'xatol': 1e-6})
            if res.success:
                refined_alpha = res.x
                roots = get_roots_from_alpha(refined_alpha)
                if roots is not None:
                    bound = compute_bound(roots)
                    if bound < best_bound:
                        best_bound = bound
                        best_roots = roots.copy()
                        best_alpha = refined_alpha
        except Exception:
            pass

    # Fallback if 1D search failed
    if best_roots is None:
        r0 = 4.0
        q = 2.0
        best_roots = np.array([r0 * (q ** i) for i in range(12)])
        best_bound = compute_bound(best_roots)

    # 12D log-space polish: refine the 1D result with a local L-BFGS step
    # This stays within the Laguerre-root family (the 1D roots are the starting point)
    # but allows small perturbations that the pure 1D family cannot express.
    if time_ok() and best_roots is not None:
        log_roots0 = np.log(best_roots)

        def cost_log(log_roots):
            roots = np.exp(log_roots)
            if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
                return 1e10
            return compute_bound(roots)

        # Tight local polish with L-BFGS-B
        try:
            res = minimize(cost_log, log_roots0, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success and res.fun < best_bound:
                new_roots = np.exp(res.x)
                if np.all(new_roots > 0) and np.all(np.isfinite(new_roots)):
                    best_bound = res.fun
                    best_roots = new_roots
        except Exception:
            pass

        # A second, even tighter polish from the improved point
        if time_ok() and best_roots is not None:
            log_roots1 = np.log(best_roots)
            try:
                res = minimize(cost_log, log_roots1, method='L-BFGS-B',
                               options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
                if res.success and res.fun < best_bound:
                    new_roots = np.exp(res.x)
                    if np.all(new_roots > 0) and np.all(np.isfinite(new_roots)):
                        best_bound = res.fun
                        best_roots = new_roots
            except Exception:
                pass

    # Ensure roots are positive and finite
    if best_roots is None:
        return [4.0 * (2.0 ** i) for i in range(12)]

    best_roots = np.maximum(best_roots, 1e-6)
    return list(best_roots)
# EVOLVE-BLOCK-END
