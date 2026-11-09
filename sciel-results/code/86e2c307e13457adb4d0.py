import time
import numpy as np
from scipy.optimize import minimize
import sys

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0

def _time_left():
    return _DEADLINE - time.time()

def _E_cpl(a, w0, wa, Om_m=0.3, Om_r=5e-5):
    """
    Compute E(a) = H(a)/H0 for CPL model.
    a is a numpy array.
    """
    Om_de = 1.0 - Om_m - Om_r
    term_m = Om_m * a**(-3)
    term_r = Om_r * a**(-4)
    cpl_factor = 1.0 + wa * (1.0 - a)
    term_de = Om_de * a**(-3.0 * (1.0 + w0 + wa)) * (cpl_factor**3)
    val = term_m + term_r + term_de
    val = np.maximum(val, 1e-30)
    return np.sqrt(val)

def _true_dL_fast(z, w0, wa, num_points=4096):
    """
    Fast reference calculation of (1+z) * integral.
    """
    if z <= 0:
        return 0.0
    zs = np.linspace(0, z, num_points)
    a_vals = 1.0 / (1.0 + zs)
    E_vals = _E_cpl(a_vals, w0, wa)
    integrand = 1.0 / E_vals
    integral = np.trapz(integrand, zs)
    return (1.0 + z) * integral

def _quad_dL(z, nodes, weights, w0, wa):
    """
    Approximate dL using the 8-node quadrature rule.
    """
    if z <= 0:
        return 0.0
    a_vals = 1.0 / (1.0 + z * nodes)
    E_vals = _E_cpl(a_vals, w0, wa)
    integral_approx = np.sum(weights / E_vals)
    return (1.0 + z) * z * integral_approx

def _worst_rel_error(nodes, weights):
    """
    Compute worst relative error across test set.
    nodes: array of 8 values in (0,1), strictly increasing
    weights: array of 8 positive values summing to 1
    """
    # Validate nodes
    if len(nodes) != 8 or len(weights) != 8:
        return 1e10
    if np.any(nodes <= 0) or np.any(nodes >= 1):
        return 1e10
    if np.any(np.diff(nodes) <= 0):
        return 1e10
    if np.any(weights <= 0):
        return 1e10
    if abs(np.sum(weights) - 1.0) > 1e-8:
        return 1e10
    
    # Broader test set covering (w0, wa) space more uniformly
    # w0 in [-2, 2], wa in [-3, 3]
    test_cosmos = [
        (0.0, 0.0),    # LCDM
        (-1.0, 0.0),   # Lambda
        (-1.0, 0.5),   # Lambda with evolution
        (-1.5, 0.5),   # Phantom
        (-1.5, -0.5),  # Phantom
        (-2.0, 1.0),   # Extreme phantom
        (-2.0, -1.0),  # Extreme phantom
        (0.0, 0.5),    # Matter-like with evolution
        (0.5, -1.0),   # Quintessence
        (0.5, 0.5),    # Quintessence
        (1.0, -2.0),   # Extreme quintessence
        (1.0, 2.0),    # Extreme quintessence
        (2.0, -3.0),   # Very extreme
        (2.0, 3.0),    # Very extreme
        (-0.5, 0.3),   # Mild
        (-0.5, -0.3),  # Mild
        (0.3, -0.3),   # Mild
        (0.3, 0.3),    # Mild
        (-1.0, -1.0),  # 
        (-1.0, 1.0),   # 
        (0.0, -1.0),   # 
        (0.0, 1.0),    # 
        (-2.5, 0.5),   # Beyond typical range
        (2.5, -0.5),   # Beyond typical range
    ]
    
    # Redshifts: cover a wide range, with emphasis on low and mid
    test_zs = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0])
    
    max_err = 0.0
    for w0, wa in test_cosmos:
        for z in test_zs:
            true_val = _true_dL_fast(z, w0, wa)
            quad_val = _quad_dL(z, nodes, weights, w0, wa)
            if true_val > 1e-10:
                rel_err = abs(quad_val - true_val) / true_val
                if rel_err > max_err:
                    max_err = rel_err
                    
    return max_err

def _decode_params(params):
    """
    Decode 16 parameters into nodes and weights.
    First 8: nodes (direct, with soft constraints)
    Last 8: log-weights (exponentiated and normalized)
    """
    # Nodes: use a transformation that ensures (0,1) and increasing
    # Map params[0:8] through sigmoid to get steps, then cumulative
    raw_nodes = params[:8]
    # Use a smooth increasing mapping
    # Shift and scale to ensure positive differences
    # Simple approach: sort and add small offsets
    nodes = np.sort(raw_nodes)
    # Ensure strictly in (0,1) with margin
    nodes = 1e-8 + (1 - 2e-8) * (nodes - nodes.min()) / max(nodes.max() - nodes.min(), 1e-12)
    # Ensure strictly increasing
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-8
    
    # Weights: exponentiate and normalize
    log_w = params[8:16]
    w_raw = np.exp(log_w - np.max(log_w))  # shift for numerical stability
    weights = w_raw / np.sum(w_raw)
    
    return nodes, weights

def _objective(params):
    """
    Objective function to minimize: worst relative error.
    """
    if _time_left() < 0.5:
        return 1e10
        
    nodes, weights = _decode_params(params)
    return _worst_rel_error(nodes, weights)

def _nelder_mead_optimize(initial_params, maxiter=200):
    """
    Run Nelder-Mead from a given initial point.
    """
    try:
        res = minimize(
            _objective, 
            initial_params, 
            method='Nelder-Mead', 
            options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        return res.x, res.fun
    except Exception:
        return initial_params, _objective(initial_params)

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 85.0
    
    best_params = None
    best_score = 1e10
    
    # Generate multiple diverse initial points
    np.random.seed(42)
    
    # Initial point 1: Gauss-Legendre nodes mapped to parameter space
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    nodes_gl = 0.5 * (x_gl + 1.0)
    weights_gl = 0.5 * w_gl
    weights_gl = weights_gl / np.sum(weights_gl)
    
    # Invert the decoding to get initial params
    # For nodes: we want raw_nodes that when sorted and normalized give nodes_gl
    # Simple: use nodes_gl directly as raw_nodes (they're already in (0,1) and sorted)
    # For weights: log of weights
    init1 = np.concatenate([nodes_gl, np.log(weights_gl)])
    
    # Initial point 2: Uniform spacing
    nodes_uniform = np.linspace(0.05, 0.95, 8)
    weights_uniform = np.ones(8) / 8
    init2 = np.concatenate([nodes_uniform, np.log(weights_uniform)])
    
    # Initial point 3: Random
    init3 = np.random.uniform(0.1, 0.9, 8)
    init3 = np.sort(init3)
    init3 = np.concatenate([init3, np.random.uniform(-1, 1, 8)])
    
    # Initial point 4: Clustered at low z
    nodes_clustered = np.array([0.02, 0.05, 0.1, 0.15, 0.25, 0.4, 0.6, 0.85])
    weights_clustered = np.array([0.1, 0.1, 0.15, 0.15, 0.2, 0.15, 0.1, 0.05])
    init4 = np.concatenate([nodes_clustered, np.log(weights_clustered)])
    
    # Initial point 5: Clustered at high z
    nodes_high = np.array([0.1, 0.2, 0.35, 0.5, 0.65, 0.75, 0.85, 0.95])
    weights_high = np.array([0.1, 0.1, 0.15, 0.2, 0.2, 0.15, 0.07, 0.03])
    init5 = np.concatenate([nodes_high, np.log(weights_high)])
    
    initial_points = [init1, init2, init3, init4, init5]
    
    # Multi-start Nelder-Mead
    for i, init in enumerate(initial_points):
        if _time_left() < 3.0:
            break
        
        # Evaluate initial point
        score = _objective(init)
        if score < best_score:
            best_score = score
            best_params = init.copy()
        
        # Run local optimization
        if _time_left() > 5.0:
            opt_params, opt_score = _nelder_mead_optimize(init, maxiter=300)
            if opt_score < best_score:
                best_score = opt_score
                best_params = opt_params.copy()
        
        # Try a second round with tighter tolerance if time permits
        if _time_left() > 8.0 and best_params is not None and np.allclose(best_params, init):
            opt_params2, opt_score2 = _nelder_mead_optimize(best_params + np.random.normal(0, 0.01, 16), maxiter=200)
            if opt_score2 < best_score:
                best_score = opt_score2
                best_params = opt_params2.copy()
    
    # Final refinement from best point
    if best_params is not None and _time_left() > 5.0:
        opt_final, score_final = _nelder_mead_optimize(best_params, maxiter=500)
        if score_final < best_score:
            best_score = score_final
            best_params = opt_final.copy()
    
    # Decode final result
    if best_params is not None:
        nodes, weights = _decode_params(best_params)
    else:
        # Fallback to Gauss-Legendre
        x_gl, w_gl = np.polynomial.legendre.leggauss(8)
        nodes = 0.5 * (x_gl + 1.0)
        weights = 0.5 * w_gl
        weights = weights / np.sum(weights)
    
    # Ensure nodes are strictly increasing and in (0,1)
    nodes = np.clip(nodes, 1e-12, 1.0 - 1e-12)
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-12
    
    # Ensure weights are positive and sum to 1
    weights = np.maximum(weights, 1e-15)
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
