"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _E_cpl(a, w0, wa, om_m, om_r, om_k=0.0):
    """E(a) = H(a)/H0 for CPL model."""
    # Ensure numerical stability
    a_safe = np.maximum(a, 1e-10)
    h2 = om_m * a_safe**(-3) + om_r * a_safe**(-4) + om_k * a_safe**0 + \
         (1 - om_m - om_r - om_k) * a_safe**(-3*(1+w0+wa))
    return np.sqrt(np.maximum(h2, 1e-30))

def _integrand_val(x, z, w0, wa, om_m, om_r, om_k):
    """The integrand 1/E(a) where a = 1/(1+z*x)."""
    a = 1.0 / (1.0 + z * x)
    return 1.0 / _E_cpl(a, w0, wa, om_m, om_r, om_k)

def _true_dL(z, w0, wa, om_m, om_r, om_k, n=5000):
    """Compute the true integral using high-resolution Simpson's rule."""
    x = np.linspace(0, 1, n)
    # Simpson's rule requires odd number of points (even number of intervals)
    if n % 2 == 0:
        n += 1
        x = np.linspace(0, 1, n)
    
    y = _integrand_val(x, z, w0, wa, om_m, om_r, om_k)
    
    # Composite Simpson's rule
    h = 1.0 / (n - 1)
    integral = h / 3.0 * (y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2]))
    
    # The formula in the task is dL/(c/H0) = (1+z)*z * integral
    # But the quadrature approximates the integral part.
    # The evaluator computes: dL_quad = (1+z)*z * sum(w_i / E(a_i))
    # The true value is: dL_true = (1+z)*z * integral
    # Relative error is the same for both.
    return integral

def _quadrature_error(params, z, w0, wa, om_m, om_r, om_k):
    """Compute relative error of quadrature vs true value for one model."""
    nodes = params[:8]
    weights = params[8:]
    
    # Ensure nodes are in [0,1] and increasing (handled by bounds/constraints in optimizer, 
    # but we check here for safety)
    if np.any(nodes < 0) or np.any(nodes > 1):
        return 1e10
    if np.any(np.diff(nodes) <= 0):
        return 1e10
    if np.any(weights <= 0):
        return 1e10
        
    a_vals = 1.0 / (1.0 + z * nodes)
    E_vals = _E_cpl(a_vals, w0, wa, om_m, om_r, om_k)
    quad_val = np.sum(weights / E_vals)
    
    true_val = _true_dL(z, w0, wa, om_m, om_r, om_k)
    
    if true_val < 1e-10:
        return abs(quad_val - true_val)
        
    return abs(quad_val - true_val) / true_val

def _loss(params):
    """Max relative error across training models."""
    # Define a diverse set of CPL models
    # (w0, wa, om_m, om_r, om_k)
    models = [
        (0.0, 0.0, 0.3, 0.05, 0.65),   # Flat Lambda CDM
        (-0.5, 0.5, 0.3, 0.05, 0.65),  # Quintessence-like
        (-1.0, 0.0, 0.25, 0.05, 0.70), # Dark Energy Dominated
        (0.5, -0.5, 0.4, 0.05, 0.55),  # Phantom-like
        (-0.9, 0.3, 0.3, 0.0, 0.70),   # Flat, no radiation
        (-1.1, 0.2, 0.2, 0.05, 0.75),  # Accelerating
    ]
    
    # Test at several redshifts
    z_vals = [0.5, 1.0, 2.0, 5.0]
    
    max_err = 0.0
    for m in models:
        w0, wa, om_m, om_r, om_k = m
        for z in z_vals:
            err = _quadrature_error(params, z, w0, wa, om_m, om_r, om_k)
            if err > max_err:
                max_err = err
    return max_err

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    # Initial guess: Gauss-Legendre
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    x0_gl = (x_gl + 1.0) / 2.0
    w0_gl = w_gl / 2.0
    x0 = np.concatenate([x0_gl, w0_gl])
    
    # Bounds: nodes in (0, 1), weights in (1e-6, 1)
    # We enforce strictly increasing nodes via the optimizer's bounds on differences?
    # No, standard minimizers don't handle inequality constraints like x_i < x_{i+1} easily without transformation.
    # Let's use a transformation or just rely on the fact that GL nodes are well spread.
    # To be safe, we can optimize over unconstrained variables and map them, 
    # but that complicates the loss function.
    # Alternative: Use L-BFGS-B with bounds on nodes [0.01, 0.99] etc, and hope they stay ordered.
    # Better: Use a parameterization that guarantees ordering.
    # Let nodes = cumulative sum of positive variables.
    # Let u_i > 0 be the gaps. sum(u_i) = 1.
    # This is complex for 16 variables.
    
    # Let's stick to direct optimization but add a penalty in the loss if ordering is violated.
    # The _quadrature_error already returns 1e10 if ordering is violated.
    
    bounds = []
    for i in range(8):
        bounds.append((1e-5, 1 - 1e-5)) # Nodes
    for i in range(8):
        bounds.append((1e-6, 1.0)) # Weights
    
    # We will run a few restarts if time permits
    best_params = x0
    best_loss = _loss(x0)
    
    # Optimization 1: L-BFGS-B
    try:
        res = minimize(_loss, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-10})
        if res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x
    except Exception as e:
        pass
    
    # Check time
    if time.time() > deadline - 5:
        # Just return best so far
        nodes = best_params[:8]
        weights = best_params[8:]
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        return {"nodes": nodes, "weights": weights}
        
    # Optimization 2: Nelder-Mead on the best result (often good for polishing)
    try:
        res2 = minimize(_loss, best_params, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
        if res2.fun < best_loss:
            best_loss = res2.fun
            best_params = res2.x
    except Exception as e:
        pass

    nodes = best_params[:8]
    weights = best_params[8:]
    
    # Final sanity check and normalization
    # Ensure nodes are strictly increasing
    if not np.all(np.diff(nodes) > 0):
        # Fallback to GL
        nodes, weights = np.polynomial.legendre.leggauss(8)
        nodes = (nodes + 1.0) / 2.0
        weights = weights / 2.0
    else:
        weights = weights / np.sum(weights)
        
    return {"nodes": nodes, "weights": weights}

# EVOLVE-BLOCK-END
