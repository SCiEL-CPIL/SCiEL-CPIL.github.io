"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_legendre
import time

def _cpl_E(z, w0, wa):
    """CPL Hubble parameter: E(z) = (1+z)^{3(1+w0+wa)/2} * (1 - (1+z)^{-3wa})^{1/(3wa)} for wa != 0"""
    z = np.asarray(z)
    if abs(wa) < 1e-10:
        return (1.0 + z) ** (3.0 * (1.0 + w0))
    # Standard CPL form
    exponent = 3.0 * (1.0 + w0 + wa) / 2.0
    base = (1.0 + z) ** exponent
    # Handle the (1 - (1+z)^{-3wa})^{1/(3wa)} term carefully
    inner = 1.0 - (1.0 + z) ** (-3.0 * wa)
    # For wa < 0 and large z, inner could be negative; clip for numerical safety
    inner = np.maximum(inner, 1e-30)
    return base * inner ** (1.0 / (3.0 * wa))

def _integrand(z, w0, wa):
    """f(z) = z / E(z) for the quadrature"""
    return z / _cpl_E(z, w0, wa)

def _quadrature_value(nodes, weights, w0, wa):
    """Compute sum_i weights[i] * f(nodes[i])"""
    return np.dot(weights, _integrand(nodes, w0, wa))

def _adaptive_reference(w0, wa, n_sub=10000):
    """High-accuracy reference using fine trapezoidal rule on [0,1]"""
    z = np.linspace(0.0, 1.0, n_sub)
    f = _integrand(z, w0, wa)
    # Composite trapezoidal
    return np.trapz(f, z)

def _log_sum_exp(x, beta=50.0):
    """Smooth approximation to max(x)"""
    x_max = np.max(x)
    if x_max < -1000:
        return 0.0
    return x_max + np.log(np.sum(np.exp(-beta * (x - x_max)))) / beta

def _objective(params):
    """
    Objective: smooth surrogate for worst-case relative error.
    params: [log_weights (8), node_transform_params (8)]
    """
    log_w = params[:8]
    node_params = params[8:16]
    
    # Transform to positive weights summing to 1
    w = np.exp(log_w)
    w = w / w.sum()
    
    # Transform to strictly increasing nodes in (0, 1)
    # Use cumulative sigmoid-like transform
    # Map node_params to (0,1) via sigmoid, then enforce ordering via cumulative
    raw_nodes = 1.0 / (1.0 + np.exp(-node_params * 3.0))  # in (0,1)
    # Ensure strictly increasing: use cumulative sum of positive differences
    # Better: sort and add small epsilon, or use a different transform
    # Use: nodes[i] = (i + 1 + raw_nodes[i]) / (n + 1) adjusted
    # Simpler: use the fact that we want them in (0,1) increasing
    # Let's use: nodes[i] = sigmoid(node_params[i]) but enforce ordering
    # Actually, let's use a cleaner transform:
    # nodes[i] = (i+1)/(n+1) + epsilon * tanh(node_params[i])
    n = 8
    base_nodes = (np.arange(n) + 1.0) / (n + 1.0)
    nodes = base_nodes + 0.05 * np.tanh(node_params)
    # Clamp to (1e-6, 1-1e-6)
    nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
    # Ensure strictly increasing
    for i in range(1, n):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-6
    nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
    
    # Compute relative errors across cosmologies
    models = [
        (0.0, 0.0),    # Lambda-CDM
        (-1.0, 0.0),   # Cosmological constant
        (-0.5, 0.0),   # Quintessence-like
        (-1.5, 0.0),   # Phantom
        (0.0, -1.0),   # CPL wa=-1
        (0.0, 1.0),    # CPL wa=1
        (-1.0, -0.5),  # CPL
        (-1.0, 0.5),   # CPL
        (0.5, -0.5),   # CPL
        (-0.5, -0.5),  # CPL
        (0.0, -0.5),   # CPL
        (-1.0, -1.0),  # CPL
    ]
    
    errors = []
    for w0, wa in models:
        ref = _adaptive_reference(w0, wa, n_sub=50000)
        if abs(ref) < 1e-15:
            errors.append(0.0)
            continue
        qval = _quadrature_value(nodes, w, w0, wa)
        rel_err = abs(qval - ref) / abs(ref)
        errors.append(rel_err)
    
    errors = np.array(errors)
    
    # Smooth max of squared errors
    return _log_sum_exp(errors ** 2, beta=100.0)

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # 75s deadline, 15s safety margin
    
    n = 8
    
    # Start from Gauss-Legendre nodes and equal weights as initial guess
    gl_nodes, gl_weights = roots_legendre(n)
    # Map from [-1,1] to [0,1]
    gl_nodes_01 = (gl_nodes + 1.0) / 2.0
    gl_weights_01 = gl_weights / 2.0
    
    # Initialize parameters
    log_w0 = np.log(gl_weights_01)
    # Invert the node transform to get initial node_params
    base_nodes = (np.arange(n) + 1.0) / (n + 1.0)
    node_params0 = np.zeros(n)
    # Try to find node_params that give gl_nodes_01
    # nodes = base_nodes + 0.05 * tanh(node_params)
    # tanh(node_params) = (gl_nodes_01 - base_nodes) / 0.05
    diffs = (gl_nodes_01 - base_nodes) / 0.05
    diffs = np.clip(diffs, -5.0, 5.0)
    node_params0 = np.arctanh(diffs)
    
    x0 = np.concatenate([log_w0, node_params0])
    
    best_params = x0.copy()
    best_obj = _objective(x0)
    
    # Run optimization
    try:
        result = minimize(
            _objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
        )
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x.copy()
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 10.0:
        # Extract and return
        log_w = best_params[:8]
        node_params = best_params[8:16]
        w = np.exp(log_w)
        w = w / w.sum()
        base_nodes = (np.arange(n) + 1.0) / (n + 1.0)
        nodes = base_nodes + 0.05 * np.tanh(node_params)
        nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
        for i in range(1, n):
            if nodes[i] <= nodes[i-1]:
                nodes[i] = nodes[i-1] + 1e-6
        nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
        return {"nodes": nodes, "weights": w}
    
    # Polish with Nelder-Mead for final refinement
    try:
        result2 = minimize(
            _objective,
            best_params,
            method='Nelder-Mead',
            options={'maxiter': 500, 'fatol': 1e-15, 'xtol': 1e-14, 'adaptive': True}
        )
        if result2.fun < best_obj:
            best_obj = result2.fun
            best_params = result2.x.copy()
    except Exception:
        pass
    
    # Extract final nodes and weights
    log_w = best_params[:8]
    node_params = best_params[8:16]
    w = np.exp(log_w)
    w = w / w.sum()
    
    base_nodes = (np.arange(n) + 1.0) / (n + 1.0)
    nodes = base_nodes + 0.05 * np.tanh(node_params)
    nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
    for i in range(1, n):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-6
    nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
    
    # Ensure weights are positive and sum to 1
    w = np.maximum(w, 1e-15)
    w = w / w.sum()
    
    return {"nodes": nodes, "weights": w}
# EVOLVE-BLOCK-END
