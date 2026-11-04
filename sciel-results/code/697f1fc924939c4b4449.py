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

    def compute_bound_and_grad(log_roots):
        """
        Compute the bound and its gradient with respect to log_roots.
        Uses automatic differentiation via finite differences for reliability.
        """
        roots = np.exp(log_roots)
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10, np.zeros(len(log_roots))
        
        bound = compute_bound(roots)
        
        # Compute gradient via central differences
        grad = np.zeros(len(log_roots))
        h = 1e-7
        for i in range(len(log_roots)):
            log_roots_plus = log_roots.copy()
            log_roots_minus = log_roots.copy()
            log_roots_plus[i] += h
            log_roots_minus[i] -= h
            
            roots_plus = np.exp(log_roots_plus)
            roots_minus = np.exp(log_roots_minus)
            
            bound_plus = compute_bound(roots_plus)
            bound_minus = compute_bound(roots_minus)
            
            grad[i] = (bound_plus - bound_minus) / (2 * h)
        
        return bound, grad

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

    # Strategy: 1D search over alpha (Laguerre-root family) + multi-start gradient polish
    best_bound = 1e10
    best_roots = None
    best_alpha = None

    # Initial coarse scan
    alpha_min = -10.0
    alpha_max = 200.0
    num_points = 300

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

        local_min = max(-10.0, best_alpha - 5.0)
        local_max = min(200.0, best_alpha + 5.0)

        try:
            res = minimize_scalar(alpha_cost, bounds=(local_min, local_max),
                                  method='bounded', options={'xatol': 1e-8})
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

    # Multi-start gradient-based polish in log-space
    if time_ok() and best_roots is not None:
        # Start from the best 1D result
        log_roots0 = np.log(best_roots)
        
        def cost_log(log_roots):
            roots = np.exp(log_roots)
            if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
                return 1e10
            return compute_bound(roots)

        def cost_log_grad(log_roots):
            bound, grad = compute_bound_and_grad(log_roots)
            return bound, grad

        # First polish from the 1D result
        try:
            res = minimize(cost_log_grad, log_roots0, method='L-BFGS-B',
                           jac=True, options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.success and res.fun < best_bound:
                new_roots = np.exp(res.x)
                if np.all(new_roots > 0) and np.all(np.isfinite(new_roots)):
                    best_bound = res.fun
                    best_roots = new_roots
        except Exception:
            pass

        # Generate diverse starting points for multi-start
        if time_ok():
            # Strategy: perturb the best roots in various ways
            start_points = []
            
            # 1. The current best
            start_points.append(np.log(best_roots))
            
            # 2. Geometric sequences with different ratios
            for r0 in [1.0, 2.0, 3.0, 5.0]:
                for q in [1.2, 1.5, 2.0, 2.5]:
                    roots = np.array([r0 * (q ** i) for i in range(12)])
                    if np.all(roots > 0):
                        start_points.append(np.log(roots))
            
            # 3. Arithmetic sequences
            for r0 in [1.0, 2.0, 3.0]:
                for d in [0.5, 1.0, 2.0, 3.0]:
                    roots = np.array([r0 + d * i for i in range(12)])
                    if np.all(roots > 0):
                        start_points.append(np.log(roots))
            
            # 4. Random perturbations of the best
            rng = np.random.default_rng(42)
            for _ in range(10):
                perturbation = rng.normal(0, 0.5, size=12)
                log_roots_pert = np.log(best_roots) + perturbation
                roots_pert = np.exp(log_roots_pert)
                if np.all(roots_pert > 0):
                    start_points.append(log_roots_pert)
            
            # Try each starting point
            for log_roots_start in start_points:
                if not time_ok():
                    break
                try:
                    res = minimize(cost_log_grad, log_roots_start, method='L-BFGS-B',
                                   jac=True, options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
                    if res.success and res.fun < best_bound:
                        new_roots = np.exp(res.x)
                        if np.all(new_roots > 0) and np.all(np.isfinite(new_roots)):
                            best_bound = res.fun
                            best_roots = new_roots
                except Exception:
                    pass

        # Final high-precision polish
        if time_ok() and best_roots is not None:
            log_roots_final = np.log(best_roots)
            try:
                res = minimize(cost_log_grad, log_roots_final, method='L-BFGS-B',
                               jac=True, options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12})
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
