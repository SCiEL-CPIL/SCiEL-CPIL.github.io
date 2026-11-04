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

    # Strategy: Direct 12D log-space optimization with multi-start
    # The objective is smooth in log-space for positive roots
    # We use multiple random initializations to avoid local minima
    best_bound = 1e10
    best_roots = None
    
    def cost_log(log_roots):
        """Compute the C4 bound from log-space root parameters."""
        if not time_ok():
            return 1e10
        roots = np.exp(log_roots)
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
        return compute_bound(roots)
    
    # Multi-start optimization in 12D log-space
    rng = np.random.RandomState(42)
    n_starts = 15
    
    for start_idx in range(n_starts):
        if not time_ok():
            break
        
        # Initialize log_roots with different strategies
        if start_idx == 0:
            # Geometric sequence starting near 4
            log_roots0 = np.log([4.0 * (1.5 ** i) for i in range(12)])
        elif start_idx == 1:
            # Geometric sequence starting near 2
            log_roots0 = np.log([2.0 * (1.8 ** i) for i in range(12)])
        elif start_idx == 2:
            # Laguerre roots as initial guess
            roots_alpha = get_roots_from_alpha(1.0)
            if roots_alpha is not None:
                log_roots0 = np.log(roots_alpha)
            else:
                log_roots0 = np.log([4.0 * (1.5 ** i) for i in range(12)])
        elif start_idx == 3:
            # Laguerre roots with alpha=5
            roots_alpha = get_roots_from_alpha(5.0)
            if roots_alpha is not None:
                log_roots0 = np.log(roots_alpha)
            else:
                log_roots0 = np.log([4.0 * (1.5 ** i) for i in range(12)])
        else:
            # Random initialization in log-space
            # Roots roughly in [1, 100]
            log_roots0 = np.sort(rng.uniform(math.log(1.0), math.log(100.0), size=12))
        
        # Add small random perturbation for diversity (except for first few structured starts)
        if start_idx >= 5:
            log_roots0 = log_roots0 + rng.normal(0, 0.1, size=12)
            log_roots0 = np.sort(log_roots0)  # Keep roots ordered
        
        # Bounds for log-space optimization
        log_bounds = [(math.log(0.1), math.log(1000.0))] * 12
        
        try:
            res = minimize(cost_log, log_roots0, method='L-BFGS-B',
                          bounds=log_bounds, 
                          options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12})
            
            if res.success and res.fun < best_bound:
                new_roots = np.exp(res.x)
                if np.all(new_roots > 0) and np.all(np.isfinite(new_roots)):
                    # Verify roots are distinct
                    if np.min(np.diff(np.sort(new_roots))) > 1e-10:
                        actual_bound = compute_bound(new_roots)
                        if actual_bound < best_bound:
                            best_bound = actual_bound
                            best_roots = new_roots
        except Exception:
            pass
    
    # Fallback if all optimizations failed
    if best_roots is None:
        # Use the Laguerre-root family as fallback
        for alpha in [0.5, 1.0, 2.0, 5.0, 10.0]:
            roots = get_roots_from_alpha(alpha)
            if roots is not None:
                bound = compute_bound(roots)
                if bound < best_bound:
                    best_bound = bound
                    best_roots = roots.copy()
        
        if best_roots is None:
            r0 = 4.0
            q = 1.5
            best_roots = np.array([r0 * (q ** i) for i in range(12)])
            best_bound = compute_bound(best_roots)

    # High-precision polish: refine with tighter tolerances
    if time_ok() and best_roots is not None:
        log_roots_polish = np.log(best_roots)
        
        # First polish pass
        try:
            res = minimize(cost_log, log_roots_polish, method='L-BFGS-B',
                           bounds=log_bounds,
                           options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14})
            if res.success and res.fun < best_bound:
                new_roots = np.exp(res.x)
                if np.all(new_roots > 0) and np.all(np.isfinite(new_roots)):
                    if np.min(np.diff(np.sort(new_roots))) > 1e-10:
                        best_bound = res.fun
                        best_roots = new_roots
        except Exception:
            pass
        
        # Second polish pass with even tighter tolerances
        if time_ok() and best_roots is not None:
            log_roots_polish2 = np.log(best_roots)
            try:
                res = minimize(cost_log, log_roots_polish2, method='L-BFGS-B',
                               bounds=log_bounds,
                               options={'maxiter': 3000, 'ftol': 1e-17, 'gtol': 1e-15})
                if res.success and res.fun < best_bound:
                    new_roots = np.exp(res.x)
                    if np.all(new_roots > 0) and np.all(np.isfinite(new_roots)):
                        if np.min(np.diff(np.sort(new_roots))) > 1e-10:
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
