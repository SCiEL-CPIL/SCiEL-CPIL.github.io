import time
import numpy as np
from scipy.optimize import minimize
import sys

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0

def _time_left():
    return _DEADLINE - time.time()

# Precompute high-order GL reference nodes and weights for fast true value calculation
_GL_ORDER = 128
_gl_nodes_1, _gl_weights_1 = np.polynomial.legendre.leggauss(_GL_ORDER)
# Map from [-1,1] to [0,1]
_GL_NODES = 0.5 * (_gl_nodes_1 + 1.0)
_GL_WEIGHTS = 0.5 * _gl_weights_1

def _E_cpl(a, w0, wa, Om_m=0.3, Om_r=5e-5):
    """
    Compute E(a) = H(a)/H0 for CPL model.
    a is a numpy array.
    """
    Om_de = 1.0 - Om_m - Om_r
    # Vectorized operations
    term_m = Om_m * a**(-3)
    term_r = Om_r * a**(-4)
    cpl_factor = 1.0 + wa * (1.0 - a)
    term_de = Om_de * a**(-3.0 * (1.0 + w0 + wa)) * (cpl_factor**3)
    val = term_m + term_r + term_de
    # Prevent division by zero or negative values due to numerical issues
    val = np.maximum(val, 1e-30)
    return np.sqrt(val)

def _true_dL_fast(z, w0, wa):
    """
    Fast reference calculation of (1+z) * integral.
    Uses high-order Gauss-Legendre quadrature for accuracy.
    """
    if z <= 0:
        return 0.0
    # Integrate over t in [0,1] where t = z'/z, so z' = z*t
    # a = 1/(1+z*t)
    a_vals = 1.0 / (1.0 + z * _GL_NODES)
    E_vals = _E_cpl(a_vals, w0, wa)
    integrand = 1.0 / E_vals
    integral = np.sum(_GL_WEIGHTS * integrand)
    return (1.0 + z) * z * integral

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

def _decode_params(params):
    """
    Decode parameters to nodes and weights.
    params: 8 node offsets from uniform, 8 log-weights.
    """
    # Nodes: start from uniform spacing and add offsets
    # Uniform nodes: (i+0.5)/8 for i=0..7
    base_nodes = (np.arange(8) + 0.5) / 8.0
    # Add small offsets (clipped to stay in (0,1) and maintain ordering)
    offsets = params[:8] * 0.05  # Scale offsets to be small
    nodes = base_nodes + offsets
    
    # Ensure strictly increasing and in (0,1)
    # Clip to valid range
    nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
    
    # Enforce strict ordering by cumulative max
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1] + 1e-12:
            nodes[i] = nodes[i-1] + 1e-12
    
    # Final check: ensure all in (0,1)
    if nodes[0] <= 0 or nodes[-1] >= 1:
        nodes = (np.arange(8) + 0.5) / 8.0
    
    # Weights: exp of params, normalized
    w_raw = np.exp(params[8:16])
    w = w_raw / np.sum(w_raw)
    
    return nodes, w

def _objective(params):
    """
    Objective function to minimize: worst relative error.
    params: 8 node offsets, 8 log-weights.
    """
    if _time_left() < 0.5:
        return 1e10
        
    x, w = _decode_params(params)
    
    # Test set - representative cosmologies
    test_cosmos = [
        (0.0, 0.0),    # LCDM
        (-1.0, 0.0),   # Lambda
        (0.5, -1.0),   # Quintessence
        (-1.5, 0.5),   # Phantom
        (-0.5, 0.5),   
        (0.3, -0.3),   
        (-2.0, 1.0),   # More extreme phantom
        (1.0, -2.0)    # More extreme quintessence
    ]
    
    # Redshifts - cover wide range
    test_zs = np.array([0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    
    max_err = 0.0
    for w0, wa in test_cosmos:
        for z in test_zs:
            true_val = _true_dL_fast(z, w0, wa)
            quad_val = _quad_dL(z, x, w, w0, wa)
            if true_val > 1e-10:
                rel_err = abs(quad_val - true_val) / true_val
                if rel_err > max_err:
                    max_err = rel_err
                    
    return max_err

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 85.0
    
    # Initial guess: uniform nodes, equal weights
    # params: 8 zeros (no offset from uniform), 8 zeros (equal log-weights)
    x0 = np.zeros(16)
    
    best_params = x0.copy()
    best_score = _objective(x0)
    
    # Nelder-Mead optimization
    if _time_left() > 30:
        try:
            res = minimize(
                _objective, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception as e:
            pass
    
    # Second refinement pass if time allows
    if _time_left() > 15:
        try:
            res2 = minimize(
                _objective, 
                best_params, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x
        except Exception as e:
            pass
    
    # Decode final result
    nodes, weights = _decode_params(best_params)
    
    # Final validation: ensure constraints
    # Nodes strictly increasing in (0,1)
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-12
    nodes = np.clip(nodes, 1e-12, 1.0 - 1e-12)
    
    # Weights positive and sum to 1
    weights = np.maximum(weights, 1e-15)
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
