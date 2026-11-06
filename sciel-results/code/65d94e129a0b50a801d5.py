"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_start_time = time.time()
DEADLINE = 85.0  # safety margin before 90s kill

def _deadline_ok():
    return (time.time() - _start_time) < DEADLINE

def _E_of_a(a, Om0, Ok0, Ol0, w0, w1):
    """H(a)/H0 for CPL model."""
    a = np.asarray(a)
    a3 = a ** 3
    a2 = a ** 2
    # w(z) = w0 + w1*z, z = 1/a - 1
    w = w0 + w1 * (1.0 / a - 1.0)
    de = Ol0 * a ** (-3.0 * (1.0 + w))
    return np.sqrt(Om0 / a3 + Ok0 / a2 + de)

def _integrand_val(t, z, Om0, Ok0, Ol0, w0, w1):
    """1/E(z*t) where a = 1/(1+z*t)"""
    a = 1.0 / (1.0 + z * t)
    return 1.0 / _E_of_a(a, Om0, Ok0, Ol0, w0, w1)

def _exact_integral(z, Om0, Ok0, Ol0, w0, w1):
    """High-precision integral of 1/E(z*t) from 0 to 1 using Gauss-Legendre 100 points."""
    # Use a high-order Gauss-Legendre rule for reference values
    # This is much faster and more consistent than quad for optimization loops
    # 100 points is overkill for accuracy but ensures stability
    x_gl, w_gl = np.polynomial.legendre.leggauss(100)
    t = 0.5 * (x_gl + 1.0)
    w_t = 0.5 * w_gl
    vals = _integrand_val(t, z, Om0, Ok0, Ol0, w0, w1)
    return np.sum(w_t * vals)

def _rel_error_vec(x_nodes, w_weights, z, Om0, Ok0, Ol0, w0, w1):
    """Compute relative error for a single cosmology."""
    a = 1.0 / (1.0 + z * x_nodes)
    E_vals = _E_of_a(a, Om0, Ok0, Ol0, w0, w1)
    approx = np.sum(w_weights / E_vals)
    exact = _exact_integral(z, Om0, Ok0, Ol0, w0, w1)
    return abs(approx - exact) / abs(exact)

def _objective(log_w_params):
    """
    Minimize the soft-max of relative errors across a grid of cosmologies.
    params: log_weights (8 values). Weights are normalized exp(log_w).
    """
    if not _deadline_ok():
        return 1e10
    
    w_raw = np.exp(log_w_params)
    w = w_raw / np.sum(w_raw)
    
    # Grid of cosmologies to optimize against
    # We use a dense grid to ensure the rule is robust
    # z in [0.1, 3.0], Om0 in [0.2, 0.4], Ok0 in [0.0, 0.1], Ol0 = 1-Om0-Ok0
    # w0 in [-1.2, -0.8], w1 in [-0.5, 0.5]
    
    # To keep computation fast, we pre-define a subset of challenging cases
    # and interpolate or just use a representative set.
    # Since the integrand is smooth, a moderate set of diverse models should suffice.
    
    z_vals = [0.2, 0.5, 1.0, 1.5, 2.0, 3.0]
    om0_vals = [0.25, 0.30, 0.35]
    ok0_vals = [0.0, 0.05]
    w0_vals = [-1.1, -1.0, -0.9]
    w1_vals = [-0.3, 0.0, 0.3]
    
    errors = []
    
    # We will iterate through a subset of these to keep the objective fast
    # But for the "universal" rule, we want to cover the space.
    # Let's use a stratified sample of ~50-100 points.
    
    # Pre-calculate exact integrals for this fixed set? 
    # No, because w0, w1 change the integrand. But we can cache if we fix the set.
    # For now, let's just compute on the fly. 100 points * 8 nodes * 100 GL points = 80k ops per eval.
    # This is very fast.
    
    for z in z_vals:
        for om0 in om0_vals:
            for ok0 in ok0_vals:
                ol0 = 1.0 - om0 - ok0
                if ol0 < 0: continue
                for w0 in w0_vals:
                    for w1 in w1_vals:
                        if not _deadline_ok():
                            return 1e10
                        err = _rel_error_vec(_x_nodes, w, z, om0, ok0, ol0, w0, w1)
                        errors.append(err)
    
    if not errors:
        return 1e10
        
    errors = np.array(errors)
    
    # Soft-max (log-sum-exp) for smooth max approximation
    # max(err) ~ beta * log(sum(exp(err/beta)))
    # As beta -> 0, it approaches max. 
    # We want to minimize the worst error, so we minimize the soft-max.
    beta = 0.01  # Small beta for sharp max, larger for smoothness
    m = np.max(errors)
    if m == 0:
        return 0.0
    soft_max = m + beta * np.log(np.sum(np.exp((errors - m) / beta)))
    
    return soft_max

def construct():
    global _x_nodes
    
    n = 8
    # Use Gauss-Legendre nodes as the fixed abscissas
    # These are optimal for integrating polynomials up to degree 2n-1
    # and generally excellent for smooth functions.
    _x_nodes, _ = np.polynomial.legendre.leggauss(n)
    
    # Map from [-1, 1] to [0, 1]
    _x_nodes = 0.5 * (_x_nodes + 1.0)
    _x_nodes = np.sort(_x_nodes)
    
    # Initial guess for weights: equal weights
    # But Gauss-Legendre weights are known. Let's start with those.
    _, w_gl = np.polynomial.legendre.leggauss(n)
    w_gl = 0.5 * w_gl # scaled to [0,1]
    w_gl = w_gl / np.sum(w_gl)
    
    log_w0 = np.log(w_gl)
    
    best_params = log_w0.copy()
    best_obj = _objective(best_params)
    
    # Optimize log-weights using L-BFGS-B
    # This is a smooth, convex-ish problem (minimizing max error)
    # L-BFGS-B is efficient for this.
    
    if _deadline_ok():
        result = minimize(
            _objective, 
            best_params, 
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x.copy()
    
    # Final polish with Nelder-Mead on the raw weights? 
    # Or just use the L-BFGS result.
    
    # Convert back to weights
    w = np.exp(best_params)
    w = w / np.sum(w)
    
    # Ensure constraints
    x = _x_nodes.copy()
    # Nodes are strictly increasing and in (0,1) by construction of leggauss
    
    # Check validity
    if np.any(w <= 0) or abs(np.sum(w) - 1.0) > 1e-12:
        # Fallback to GL weights if something went wrong
        w = w_gl.copy()
        
    return {"nodes": x, "weights": w}
# EVOLVE-BLOCK-END
