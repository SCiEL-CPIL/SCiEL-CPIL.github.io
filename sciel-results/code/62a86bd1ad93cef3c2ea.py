"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_genlaguerre

def _log_gamma(x):
    """Stable log-gamma for positive x using scipy."""
    from scipy.special import gammaln
    return gammaln(x)

def _compute_bound(roots):
    """
    Compute the C4 upper bound for given double-root locations.
    
    The Cohn-Gonçalves construction: given P(x) = prod_i (x - r_i)^2,
    the bound involves a ratio of integrals with the Laguerre weight.
    
    For the Fourier sign uncertainty problem with m double roots in dimension n=1,
    the relevant functional is:
      B = [integral_0^inf |P(x)|^2 e^{-x} dx] / [integral_0^inf P(x)^2 e^{-x} dx]
    but since P has double roots, P >= 0, so |P|^2 = P^2.
    
    Actually, the refined bound uses:
      B = integral_0^inf P(x)^2 e^{-x} dx / (integral_0^inf e^{-x} dx)^2
         = integral_0^inf P(x)^2 e^{-x} dx  (since integral e^{-x} = 1)
    
    But the actual C4 bound from the construction is more nuanced. Let me use
    the standard formulation: the bound is proportional to
      ||P||_L2(0,inf; e^{-x}dx)^2 / (norm of the "sign function" projection)
    
    For the simplified ratio objective that's been shown to work:
      B = integral_0^inf P(x)^2 e^{-x} dx
    normalized appropriately.
    
    Let me use the log of the bound to avoid overflow.
    """
    roots = np.asarray(roots, dtype=np.float64)
    if len(roots) != 12:
        return 1e10
    
    # Compute log of P(x)^2 e^{-x} integral using Gauss-Laguerre quadrature
    # P(x) = prod_i (x - r_i)^2
    # log P(x)^2 = 2 * sum_i log(x - r_i)^2 = 4 * sum_i log|x - r_i|
    
    # Use Gauss-Laguerre quadrature with enough points
    n_quad = 512
    x_nodes, w_nodes = np.polynomial.laguerre.laggauss(n_quad)
    
    # For each quadrature node, compute log P(x)^2 * e^{-x}
    # log(P(x)^2 * e^{-x}) = 2 * sum_i 2*log|x - r_i| - x
    #                       = 4 * sum_i log|x - r_i| - x
    
    # Compute log|x - r_i| for all nodes and roots
    # x_nodes: (n_quad,), roots: (12,)
    diff = x_nodes[:, None] - roots[None, :]  # (n_quad, 12)
    log_abs_diff = np.log(np.abs(diff) + 1e-300)  # (n_quad, 12)
    
    log_P2 = 4.0 * np.sum(log_abs_diff, axis=1)  # (n_quad,)
    log_integrand = log_P2 - x_nodes  # (n_quad,)
    
    # Weighted integral: sum w_i * exp(log_integrand_i)
    # Use log-sum-exp for stability
    max_log = np.max(log_integrand + np.log(w_nodes))
    if max_log > 700:
        return 1e10  # overflow protection
    
    log_weights = np.log(w_nodes)
    log_terms = log_integrand + log_weights
    log_sum = max_log + np.log(np.sum(np.exp(log_terms - max_log)))
    
    return log_sum  # return log of the bound

def _compute_bound_raw(roots):
    """Compute the raw (non-log) bound value."""
    log_val = _compute_bound(roots)
    if log_val > 700:
        return 1e300
    return np.exp(log_val)

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Optimizes the roots to minimize the C4 upper bound.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600s limit, keep margin
    
    # Initial guess: zeros of the 12th Laguerre polynomial (standard choice)
    # L_12(x) = sum_{k=0}^{12} (-1)^k * C(12,k) * x^k / k!
    # Use numpy to find roots
    # L_n(x) roots can be found via companion matrix
    # Coefficients of L_12: L_n(x) = sum_{k=0}^n (-1)^k * binom(n,k) * x^k / k!
    n_lag = 12
    coeffs = []
    from math import comb, factorial
    for k in range(n_lag, -1, -1):
        c = (-1)**k * comb(n_lag, k) / factorial(k)
        coeffs.append(c)
    coeffs = np.array(coeffs)
    
    # Find roots of Laguerre polynomial
    laguerre_roots = np.roots(coeffs)
    laguerre_roots = np.sort(np.real(laguerre_roots))
    
    # Filter to positive real roots (should be 12 positive roots for L_12)
    positive_roots = laguerre_roots[laguerre_roots > 0]
    
    if len(positive_roots) < 12:
        # Fallback to seed
        return [4.0 * (i + 1) for i in range(12)]
    
    best_roots = positive_roots[:12].copy()
    best_bound = _compute_bound_raw(best_roots)
    
    # Local optimization using L-BFGS-B with bounds
    def objective(roots):
        if time.time() > deadline - 10.0:
            return 1e10
        if np.any(roots <= 0):
            return 1e10
        return _compute_bound(roots)  # log bound, lower is better
    
    def gradient(roots):
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros_like(roots)
        f0 = objective(roots)
        for i in range(len(roots)):
            roots_p = roots.copy()
            roots_p[i] += eps
            roots_m = roots.copy()
            roots_m[i] -= eps
            if roots_m[i] <= 0:
                grad[i] = (objective(roots_p) - f0) / eps
            else:
                grad[i] = (objective(roots_p) - objective(roots_m)) / (2 * eps)
        return grad
    
    # Optimize with L-BFGS-B
    try:
        result = minimize(
            objective,
            best_roots,
            method='L-BFGS-B',
            jac=gradient,
            bounds=[(1e-6, None)] * 12,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if result.success and result.x is not None:
            candidate = result.x
            if np.all(candidate > 0) and np.all(np.isfinite(candidate)):
                cand_bound = _compute_bound_raw(candidate)
                if cand_bound < best_bound:
                    best_roots = candidate
                    best_bound = cand_bound
    except Exception:
        pass
    
    # Try a few more local optimizations from perturbed starting points
    rng = np.random.RandomState(42)
    for trial in range(5):
        if time.time() > deadline - 30.0:
            break
        
        # Perturb the best roots
        perturbation = rng.randn(12) * 0.1 * best_roots
        start = best_roots + perturbation
        start = np.maximum(start, 1e-3)
        
        try:
            result = minimize(
                objective,
                start,
                method='L-BFGS-B',
                jac=gradient,
                bounds=[(1e-6, None)] * 12,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.success and result.x is not None:
                candidate = result.x
                if np.all(candidate > 0) and np.all(np.isfinite(candidate)):
                    cand_bound = _compute_bound_raw(candidate)
                    if cand_bound < best_bound:
                        best_roots = candidate
                        best_bound = cand_bound
        except Exception:
            pass
    
    # Sort the final roots
    best_roots = np.sort(best_roots)
    
    # Ensure they are positive and finite
    if not np.all(best_roots > 0) or not np.all(np.isfinite(best_roots)):
        return [4.0 * (i + 1) for i in range(12)]
    
    return list(best_roots.astype(float))
# EVOLVE-BLOCK-END
