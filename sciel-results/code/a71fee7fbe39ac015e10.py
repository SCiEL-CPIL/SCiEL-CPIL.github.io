"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_genlaguerre, genlaguerre
from scipy.optimize import minimize_scalar, minimize
import time

def _compute_c4_bound(roots, m=12, n=1):
    """
    Compute the C4 upper bound from the Laguerre polynomial construction.
    The polynomial P has double roots at the given locations.
    The bound involves the L2 norm with respect to the Laguerre weight e^{-x} x^n.
    """
    roots = np.array(roots, dtype=float)
    if len(roots) != m:
        return 1e10
    if np.any(roots <= 0):
        return 1e10
    
    # Build the polynomial with double roots: P(x) = prod_i (x - r_i)^2
    # Evaluate the L2 norm: integral of |P(x)|^2 * e^{-x} * x^n dx
    # Use Gauss-Laguerre quadrature for the integral
    # We need enough quadrature points to integrate a polynomial of degree 2*m = 24
    # times x^n, so degree 24+n. Use n_quad = 2*m + n + 20 to be safe.
    n_quad = 2 * m + n + 30
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # Evaluate P(x_quad) = prod_i (x_quad - r_i)^2
    # Use log-sum for numerical stability
    log_P = np.zeros_like(x_quad)
    for r in roots:
        log_P += 2.0 * np.log(np.abs(x_quad - r) + 1e-30)
    # P(x) is positive everywhere since all terms are squared
    P_vals = np.exp(log_P)
    
    # Weight: e^{-x} * x^n
    weight = np.exp(-x_quad) * x_quad**n
    integrand = P_vals**2 * weight
    
    # Integrate
    norm_sq = np.sum(w_quad * integrand)
    
    if norm_sq <= 0 or not np.isfinite(norm_sq):
        return 1e10
    
    # The C4 bound is inversely related to the norm
    # A common form: C4 ~ 1 / sqrt(norm_sq) or similar
    # Let's use 1/sqrt(norm_sq) as the bound
    bound = 1.0 / np.sqrt(norm_sq)
    
    return bound

def _gauss_laguerre_nodes(m, alpha):
    """
    Get the nodes (roots) of the generalized Laguerre polynomial L_m^(alpha)(x).
    These are the Gauss-Laguerre quadrature nodes for weight e^{-x} x^alpha.
    """
    # Use scipy's genlaguerre to get the polynomial, then find roots
    try:
        # L_m^(alpha) has m positive roots
        # Use numpy polynomial companion matrix method
        coeffs = genlaguerre(m, alpha)
        # coeffs[0] is leading coefficient
        roots = np.roots(coeffs)
        # Keep only real positive roots
        roots = roots[np.isreal(roots)]
        roots = np.real(roots)
        roots = roots[roots > 0]
        roots = np.sort(roots)
        if len(roots) == m:
            return roots
    except Exception:
        pass
    
    # Fallback: use Newton's method or other approach
    # Use the fact that roots are positive and interlaced
    # Simple fallback: uniform spacing
    return np.linspace(1.0, m * 2.0, m)

def construct():
    m = 12
    n = 1
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second limit
    
    # Strategy: 1D search over alpha for generalized Gauss-Laguerre nodes
    # alpha controls the density profile of the roots
    # For alpha = 0: standard Laguerre nodes (clustered near 0)
    # For large alpha: nodes spread out more
    
    best_score = np.inf
    best_roots = None
    
    # Initial guess from seed: uniform roots 4, 8, ..., 48
    seed_roots = np.array([4.0 * (i + 1) for i in range(12)])
    seed_score = _compute_c4_bound(seed_roots, m, n)
    best_score = seed_score
    best_roots = seed_roots.copy()
    
    # Search over alpha
    # The seed roots are roughly uniform from 4 to 48
    # Standard Laguerre nodes (alpha=0) for m=12 are clustered near 0
    # We need alpha large enough to spread them out
    # Let's search alpha in a reasonable range
    
    def objective_alpha(alpha):
        if time.time() > deadline - 30:
            return 1e10
        try:
            roots = _gauss_laguerre_nodes(m, alpha)
            if roots is None or len(roots) != m:
                return 1e10
            if np.any(roots <= 0):
                return 1e10
            return _compute_c4_bound(roots, m, n)
        except Exception:
            return 1e10
    
    # Coarse grid search over alpha
    alpha_values = np.linspace(-0.5, 30.0, 100)
    for alpha in alpha_values:
        if time.time() > deadline - 60:
            break
        score = objective_alpha(alpha)
        if score < best_score:
            best_score = score
            best_roots = _gauss_laguerre_nodes(m, alpha).copy()
    
    # Fine search around the best alpha
    if best_roots is not None and time.time() < deadline - 60:
        # Find the alpha that gave the best score
        # Re-do a finer search
        # First, find approximate best alpha from coarse grid
        best_alpha = alpha_values[0]
        best_coarse_score = 1e10
        for alpha in alpha_values:
            score = objective_alpha(alpha)
            if score < best_coarse_score:
                best_coarse_score = score
                best_alpha = alpha
        
        # Fine search around best_alpha
        alpha_fine = np.linspace(max(-0.5, best_alpha - 5.0), best_alpha + 5.0, 50)
        for alpha in alpha_fine:
            if time.time() > deadline - 30:
                break
            score = objective_alpha(alpha)
            if score < best_score:
                best_score = score
                best_roots = _gauss_laguerre_nodes(m, alpha).copy()
    
    # Now do local optimization on the roots themselves
    # Starting from the best Gauss-Laguerre nodes
    if best_roots is not None and time.time() < deadline - 30:
        x0 = best_roots.copy()
        
        def objective_roots(params):
            if time.time() > deadline - 10:
                return 1e10
            try:
                roots = np.sort(np.abs(params))
                if np.any(roots < 1e-6):
                    return 1e10
                return _compute_c4_bound(roots, m, n)
            except Exception:
                return 1e10
        
        try:
            res = minimize(objective_roots, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6,
                                   'adaptive': True})
            opt_roots = np.sort(np.abs(res.x))
            if np.all(opt_roots > 0) and len(opt_roots) == m:
                opt_score = _compute_c4_bound(opt_roots, m, n)
                if opt_score < best_score:
                    best_score = opt_score
                    best_roots = opt_roots.copy()
        except Exception:
            pass
    
    # Final check
    if best_roots is None:
        best_roots = seed_roots
    
    return list(best_roots)
# EVOLVE-BLOCK-END
