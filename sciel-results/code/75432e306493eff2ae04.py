"""Refined Cohn-Goncalves Laguerre construction for Fourier sign uncertainty."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import roots_genlaguerre, eval_genlaguerre
from scipy.optimize import minimize

def _compute_c4(roots):
    """Compute the C4 upper bound = 1/sqrt(prod(roots))."""
    if len(roots) != 12:
        return float('inf')
    if np.any(roots <= 0):
        return float('inf')
    prod = np.prod(roots)
    if prod <= 0 or not np.isfinite(prod):
        return float('inf')
    return 1.0 / np.sqrt(prod)

def _log_c4(roots):
    """Compute log(C4) = -0.5 * sum(log(roots))."""
    if len(roots) != 12:
        return float('inf')
    if np.any(roots <= 0):
        return float('inf')
    return -0.5 * np.sum(np.log(roots))

def _laguerre_residual(roots, m=12, n=1):
    """
    Compute the Laguerre constraint residual.
    For the Cohn-Goncalves construction, the polynomial
    P(x) = prod_i (x - r_i)^2 must satisfy that
    the Laguerre combination is well-defined.
    
    The key constraint: P(x) * e^{-x} * x^n should be orthogonal
    to lower-degree polynomials in the Laguerre weight space.
    
    We check that the moments match: for k=0,...,m-1,
    integral_0^inf P(x) x^{k+n} e^{-x} dx should equal
    the corresponding value for the ideal construction.
    """
    m = 12
    n = 1
    # The constraint is that P(x) = prod (x-r_i)^2 should be such that
    # the generalized Laguerre polynomial condition holds.
    # A practical check: the roots should be close to zeros of L_m^(m+n-1)(x)
    # or satisfy the moment conditions.
    
    # Compute P(x) coefficients
    # P(x) = prod_{i=1}^{12} (x - r_i)^2
    # = [prod (x - r_i)]^2
    roots_arr = np.array(roots)
    # Compute Q(x) = prod (x - r_i)
    q_coeffs = np.poly(roots_arr)  # monic polynomial with these roots
    # P(x) = Q(x)^2
    p_coeffs = np.convolve(q_coeffs, q_coeffs)
    
    # The Laguerre constraint: for the Cohn-Goncalves construction with m=12, n=1,
    # we need that P(x) is a "Laguerre polynomial" in the sense that
    # P(x) e^{-x} x^n is orthogonal to x^k e^{-x} x^n for k < m
    # i.e., integral_0^inf P(x) x^{k+n} e^{-x} dx = 0 for k = 0, ..., m-1
    
    # Actually, the constraint is more subtle. Let me use the fact that
    # the optimal roots are the zeros of L_m^{(alpha)}(x) for appropriate alpha.
    # For m=12, n=1, the relevant parameter is alpha = m + n - 1 = 12.
    
    # Residual: check how well the roots match the zeros of L_12^(12)(x)
    target_roots, _ = roots_genlaguerre(m, m)  # L_m^(m)
    
    # Sort both
    roots_sorted = np.sort(roots_arr)
    target_sorted = np.sort(target_roots)
    
    # Compute relative residual
    residual = np.sum((roots_sorted - target_sorted)**2) / np.sum(target_sorted**2)
    
    return residual

def _constrained_objective(x, m=12, n=1):
    """
    Objective for optimization: minimize C4 (maximize product) subject to
    Laguerre constraint.
    Uses log-space to avoid overflow.
    """
    roots = np.exp(x)  # ensure positivity via exp transform
    c4 = _log_c4(roots)
    if not np.isfinite(c4):
        return 1e10
    return c4

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    deadline = time.time() + 580.0  # safety margin before 630s kill
    
    # Start with the zeros of L_12^(12)(x) - the natural Cohn-Goncalves roots
    m = 12
    n = 1
    alpha = m + n - 1  # = 12
    
    # Get zeros of generalized Laguerre polynomial L_m^(alpha)(x)
    roots_lag, weights = roots_genlaguerre(m, alpha)
    
    # Sort roots
    roots = np.sort(roots_lag)
    
    # Verify they are all positive
    if np.any(roots <= 0):
        # Fallback: use simple positive roots
        roots = np.array([4.0 * (i + 1) for i in range(12)])
        return list(roots)
    
    initial_c4 = _compute_c4(roots)
    initial_log_c4 = _log_c4(roots)
    
    # Try to optimize: maximize product (minimize log C4) 
    # while keeping roots close to the Laguerre zeros (constraint satisfaction)
    # Use the log-space parameterization: roots = exp(x)
    
    x0 = np.log(roots.copy())
    
    # Define a composite objective: minimize log_c4 + penalty * constraint_residual
    # The constraint ensures we stay in the valid Laguerre construction family
    
    def composite_objective(x):
        roots_local = np.exp(x)
        log_c4_val = _log_c4(roots_local)
        if not np.isfinite(log_c4_val):
            return 1e10
        # Constraint: roots should be close to Laguerre zeros
        # Use a soft constraint: penalty for deviation
        roots_sorted = np.sort(roots_local)
        target_sorted = np.sort(roots_lag)
        # Relative deviation
        dev = np.sum((roots_sorted - target_sorted)**2) / np.sum(target_sorted**2)
        # Penalty weight: strong enough to keep constraint satisfied
        penalty_weight = 100.0
        return log_c4_val + penalty_weight * dev
    
    best_x = x0.copy()
    best_obj = composite_objective(x0)
    
    # Try local optimization
    try:
        result = minimize(composite_objective, x0, method='L-BFGS-B', 
                         options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12})
        if result.success and result.fun < best_obj:
            best_x = result.x
            best_obj = result.fun
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 30:
        roots_final = np.exp(best_x)
        if np.any(roots_final <= 0) or not np.all(np.isfinite(roots_final)):
            roots_final = roots.copy()
        return list(roots_final)
    
    # Try a second optimization with different starting points
    # Perturb the initial roots and try again
    np.random.seed(42)
    for trial in range(5):
        if time.time() > deadline - 60:
            break
        
        # Perturb
        x_trial = x0 + np.random.randn(12) * 0.1
        try:
            result = minimize(composite_objective, x_trial, method='L-BFGS-B',
                             options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12})
            if result.success and result.fun < best_obj:
                best_x = result.x
                best_obj = result.fun
        except Exception:
            pass
    
    # Final roots
    roots_final = np.exp(best_x)
    
    # Ensure positivity and finiteness
    if np.any(roots_final <= 0) or not np.all(np.isfinite(roots_final)):
        roots_final = roots.copy()
    
    # Final check: compute C4
    final_c4 = _compute_c4(roots_final)
    if not np.isfinite(final_c4) or final_c4 >= initial_c4:
        # If optimization didn't help, use the original Laguerre roots
        roots_final = roots.copy()
    
    return list(roots_final)
# EVOLVE-BLOCK-END
