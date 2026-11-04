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

    # Strategy: Rational potential parameterization with K=3 poles
    # V(x) = sum_{k=1}^{K} a_k / (x - c_k), K=3
    # Stationarity: for each root x_i, sum_{j != i} 1/(x_i - x_j) + V'(x_i) = 0
    # This reduces the search from 12 independent coordinates to 9 coupled parameters
    # (3 poles c_k + 3 coefficients a_k + 3 shape parameters for root placement)
    best_bound = 1e10
    best_roots = None

    def get_roots_from_potential(params):
        """
        Given potential parameters, find 12 positive roots satisfying stationarity.
        params = [c1, c2, c3, a1, a2, a3, log_scale, log_spacing, log_curvature]
        
        The roots are placed as a smooth family:
        x_i = scale * (1 + spacing * i + curvature * i^2 / 12) for i=0..11
        Then we solve for the potential that makes them stationary,
        or use the potential to define a cost for optimization.
        """
        if len(params) != 9:
            return None
        c1, c2, c3, a1, a2, a3, log_scale, log_spacing, log_curvature = params
        
        scale = np.exp(log_scale)
        spacing = np.exp(log_spacing)
        curvature = np.exp(log_curvature)
        
        # Generate 12 root positions
        i = np.arange(12, dtype=float)
        roots = scale * (1.0 + spacing * i + curvature * i * i / 12.0)
        
        if np.any(roots <= 0) or np.any(~np.isfinite(roots)):
            return None
        
        # Check that roots are strictly increasing (positive spacing)
        if np.any(np.diff(roots) <= 1e-12):
            return None
        
        return roots

    def compute_stationarity_residual(roots, params):
        """
        Compute the stationarity residual:
        For each root x_i: sum_{j != i} 1/(x_i - x_j) + V'(x_i) = 0
        where V'(x) = -sum_k a_k / (x - c_k)^2
        """
        c1, c2, c3, a1, a2, a3, _, _, _ = params
        poles = np.array([c1, c2, c3])
        coeffs = np.array([a1, a2, a3])
        
        residual = 0.0
        for i in range(12):
            xi = roots[i]
            # Sum over other roots
            s = 0.0
            for j in range(12):
                if j != i:
                    s += 1.0 / (xi - roots[j])
            
            # V'(xi) = -sum_k a_k / (xi - c_k)^2
            vp = 0.0
            for k in range(3):
                vp -= coeffs[k] / (xi - poles[k])**2
            
            residual += (s + vp)**2
        
        return residual

    def potential_cost(params):
        """
        Cost function: stationarity residual + bound on C4
        We want roots that are both stationary (physically meaningful)
        and give a good bound.
        """
        if not time_ok():
            return 1e10
        
        roots = get_roots_from_potential(params)
        if roots is None:
            return 1e10
        
        # Check that potential poles don't coincide with roots
        for k in range(3):
            c = params[k]
            if np.min(np.abs(roots - c)) < 1e-6:
                return 1e10
        
        # Stationarity residual
        res = compute_stationarity_residual(roots, params)
        
        # Bound on C4
        bound = compute_bound(roots)
        
        # Combined cost: weight the residual heavily if it's large
        if res > 1e-4:
            return 1e6 * res + bound
        
        return bound + 1e-6 * res

    # Multi-start optimization over the 9-dimensional parameter space
    n_starts = 8
    rng = np.random.RandomState(42)
    
    for start_idx in range(n_starts):
        if not time_ok():
            break
        
        # Initialize parameters
        if start_idx == 0:
            # Start near a reasonable geometric sequence
            log_scale = math.log(4.0)
            log_spacing = math.log(1.5)
            log_curvature = math.log(0.1)
        else:
            log_scale = math.log(rng.uniform(1.0, 20.0))
            log_spacing = math.log(rng.uniform(0.5, 3.0))
            log_curvature = math.log(rng.uniform(0.01, 1.0))
        
        # Poles: place them between or outside the root cluster
        # Roots are roughly in [scale, scale*(1+11*spacing+11*curvature)]
        scale_est = np.exp(log_scale)
        spread_est = scale_est * (1 + 11 * np.exp(log_spacing))
        
        c1 = scale_est * 0.5
        c2 = scale_est * 1.5
        c3 = scale_est * 2.5
        
        # Coefficients: start with equal weights
        a1 = scale_est * 0.1
        a2 = scale_est * 0.1
        a3 = scale_est * 0.1
        
        params0 = np.array([c1, c2, c3, a1, a2, a3, 
                           log_scale, log_spacing, log_curvature])
        
        # Bounds for optimization
        bounds = [
            (scale_est * 0.1, scale_est * 3.0),   # c1
            (scale_est * 0.5, scale_est * 3.5),   # c2
            (scale_est * 1.0, scale_est * 4.0),   # c3
            (-scale_est, scale_est),              # a1
            (-scale_est, scale_est),              # a2
            (-scale_est, scale_est),              # a3
            (math.log(0.5), math.log(50.0)),      # log_scale
            (math.log(0.3), math.log(5.0)),       # log_spacing
            (math.log(0.01), math.log(2.0)),      # log_curvature
        ]
        
        try:
            res = minimize(potential_cost, params0, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if res.success and res.fun < best_bound:
                new_roots = get_roots_from_potential(res.x)
                if new_roots is not None and np.all(new_roots > 0) and np.all(np.isfinite(new_roots)):
                    actual_bound = compute_bound(new_roots)
                    if actual_bound < best_bound:
                        best_bound = actual_bound
                        best_roots = new_roots
        except Exception:
            pass
    
    # Fallback if potential-based search failed
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
            q = 2.0
            best_roots = np.array([r0 * (q ** i) for i in range(12)])
            best_bound = compute_bound(best_roots)

    # 12D log-space polish: refine the potential-based result with a local L-BFGS step
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
                           options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
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
                               options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12})
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
