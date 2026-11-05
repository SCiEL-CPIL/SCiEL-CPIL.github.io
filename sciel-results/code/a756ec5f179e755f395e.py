"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import trapezoid
import time

def _E_squared(a, Om, w0, wa):
    """Vectorized E^2(a) for CPL model."""
    Ode = 1.0 - Om
    if Ode < 0:
        Ode = 0.0
    term1 = Om / (a**3)
    term2 = Ode * (a**(-3.0 * w0)) * np.exp(-3.0 * wa * (1.0 - a))
    return np.maximum(term1 + term2, 1e-30)

def _quadrature_value(nodes, weights, z, Om, w0, wa):
    """Compute quadrature approximation for luminosity distance."""
    a_vals = 1.0 / (1.0 + z * nodes)
    E2 = _E_squared(a_vals, Om, w0, wa)
    E = np.sqrt(E2)
    return (1.0 + z) * z * np.dot(weights, 1.0 / E)

def _reference_value_fast(z, Om, w0, wa, n_ref=2000):
    """High-accuracy reference using vectorized trapezoidal rule on t in [0,1]."""
    if z < 1e-12:
        return 0.0
    t_grid = np.linspace(0, 1, n_ref)
    a_vals = 1.0 / (1.0 + z * t_grid)
    E2 = _E_squared(a_vals, Om, w0, wa)
    E = np.sqrt(E2)
    f = 1.0 / E
    return trapezoid(f, t_grid)

def _compute_errors(nodes, weights, portfolio, z_grid, n_ref=1500):
    """Compute relative errors for given nodes/weights across portfolio and z_grid."""
    errors = []
    for Om, w0, wa in portfolio:
        for z in z_grid:
            quad_val = _quadrature_value(nodes, weights, z, Om, w0, wa)
            ref_val = _reference_value_fast(z, Om, w0, wa, n_ref=n_ref)
            if abs(ref_val) > 1e-10:
                rel_err = abs(quad_val - ref_val) / abs(ref_val) * 100.0
                errors.append(rel_err)
    return np.array(errors) if errors else np.array([1e10])

def _soft_max(errors, beta=50.0):
    """Smooth max via log-sum-exp."""
    errors = np.clip(errors, 0, 1e6)
    max_e = np.max(errors)
    if max_e < 1e-10:
        return max_e
    shifted = errors - max_e
    log_sum_exp = max_e + np.log(np.sum(np.exp(beta * shifted)))
    return (1.0 / beta) * log_sum_exp

def _logit_transform(params):
    """Transform from unconstrained params to valid nodes and weights.
    
    params: [theta_1..theta_7, log_w_1..log_w_8]
    theta_i define node positions via cumulative softmax
    log_w_i define weights via softmax
    """
    # Nodes: use cumulative sum of softmax of theta to ensure strictly increasing in (0,1)
    theta = params[:7]
    # Add a fixed first and last "node" to frame the interval
    # We want 8 nodes in (0,1), strictly increasing
    # Use 7 interior boundaries via softmax
    # Actually, let's use a different approach:
    # nodes[i] = sigmoid(theta[i]) for i=0..7, then sort and enforce strict increase
    # Better: use cumulative softmax of 8 values, but we need exactly 8 nodes
    
    # Let's use: 7 boundary parameters, nodes are cumulative
    # Actually simplest: use 8 parameters for nodes via sigmoid, then sort
    # But we need strict increase. Let's use the cumulative approach.
    
    # Use 7 parameters for 7 gaps, nodes = cumulative sum of softmax(gaps) normalized
    # This ensures nodes in (0,1) and strictly increasing
    
    # Actually, let me use a cleaner approach:
    # 8 node params -> sigmoid -> sort -> enforce min spacing
    # 8 weight params -> softmax -> weights
    
    node_params = params[:8]
    weight_params = params[8:16]
    
    # Nodes via sigmoid
    nodes = 1.0 / (1.0 + np.exp(-node_params * 5.0))  # scale for better conditioning
    nodes = np.sort(nodes)
    
    # Enforce strict increase with minimum spacing
    min_spacing = 1e-4
    for i in range(1, 8):
        if nodes[i] - nodes[i-1] < min_spacing:
            nodes[i] = nodes[i-1] + min_spacing
    # Clamp to (0, 1)
    nodes = np.clip(nodes, 1e-6, 1 - 1e-6)
    # Final strict increase check
    for i in range(1, 8):
        if nodes[i] - nodes[i-1] < 1e-10:
            nodes[i] = nodes[i-1] + 1e-10
    nodes = np.clip(nodes, 1e-8, 1 - 1e-8)
    
    # Weights via softmax
    weight_params_shifted = weight_params - np.max(weight_params)
    exp_w = np.exp(weight_params_shifted)
    weights = exp_w / np.sum(exp_w)
    
    return nodes, weights

def _objective_constrained(x, portfolio, z_grid, t_start, deadline, n_ref=1500):
    """Objective using soft-max smoothing with logit parameterization."""
    if time.time() - t_start > deadline:
        return 1e10
    
    nodes, weights = _logit_transform(x)
    
    errors = _compute_errors(nodes, weights, portfolio, z_grid, n_ref=n_ref)
    return _soft_max(errors, beta=50.0)

def construct():
    t_start = time.time()
    deadline = 85.0  # seconds, leave 5s margin
    
    # Portfolio of cosmologies to optimize for - diverse CPL models
    portfolio = [
        (0.30, -1.0, 0.0),   # Lambda CDM
        (0.30, -1.0, 0.3),   # CPL with positive wa
        (0.30, -1.0, -0.3),  # CPL with negative wa
        (0.27, -1.0, 0.0),
        (0.27, -1.2, 0.5),
        (0.27, -0.8, -0.5),
        (0.31, -1.1, 0.2),
        (0.31, -0.9, -0.2),
        (0.30, -0.5, 0.3),   # Matter-dominated-like
        (0.30, -1.5, 0.0),   # Phantom-like
    ]
    
    z_grid = np.linspace(0.1, 2.0, 30)
    
    # Initial point: Gauss-Legendre
    nodes_gl, weights_gl = np.polynomial.legendre.leggauss(8)
    nodes_gl = (nodes_gl + 1.0) / 2.0
    weights_gl = weights_gl / 2.0
    
    # Invert logit transform to get initial params
    # For nodes: theta = logit(node) / 5
    initial_nodes = nodes_gl.copy()
    initial_weights = weights_gl.copy()
    
    # Compute initial params
    # Nodes: node = sigmoid(5*theta) => theta = logit(node)/5
    node_params = np.log(initial_nodes / (1 - initial_nodes)) / 5.0
    # Weights: w = softmax(log_w) => log_w = log(w) (up to constant)
    weight_params = np.log(initial_weights + 1e-30)
    
    x0 = np.concatenate([node_params, weight_params])
    
    best_x = x0.copy()
    best_obj = _objective_constrained(x0, portfolio, z_grid, t_start, deadline)
    
    # Phase 1: Differential evolution for global exploration
    if time.time() - t_start < deadline - 20.0:
        try:
            bounds = [(-5, 5)] * 8 + [(-5, 5)] * 8
            de_result = differential_evolution(
                lambda x: _objective_constrained(x, portfolio, z_grid, t_start, deadline),
                bounds=bounds,
                maxiter=30,
                popsize=15,
                tol=1e-4,
                seed=42,
                polish=False,
                init='sobol',
                updating='immediate'
            )
            if de_result.fun < best_obj:
                best_obj = de_result.fun
                best_x = de_result.x
        except Exception:
            pass
    
    # Phase 2: SLSQP local polish
    if time.time() - t_start < deadline - 5.0:
        try:
            slsqp_result = minimize(
                lambda x: _objective_constrained(x, portfolio, z_grid, t_start, deadline),
                best_x,
                method='SLSQP',
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            if slsqp_result.fun < best_obj:
                best_obj = slsqp_result.fun
                best_x = slsqp_result.x
        except Exception:
            pass
    
    # Phase 3: Final fine polish with more z points
    if time.time() - t_start < deadline - 2.0:
        try:
            z_grid_fine = np.linspace(0.05, 2.5, 50)
            fine_result = minimize(
                lambda x: _objective_constrained(x, portfolio, z_grid_fine, t_start, deadline, n_ref=2000),
                best_x,
                method='SLSQP',
                options={'maxiter': 100, 'ftol': 1e-14}
            )
            if fine_result.fun < best_obj:
                best_obj = fine_result.fun
                best_x = fine_result.x
        except Exception:
            pass
    
    # Extract final nodes and weights
    nodes, weights = _logit_transform(best_x)
    
    # Ensure constraints are met
    weights = np.maximum(weights, 1e-10)
    weights = weights / np.sum(weights)
    nodes = np.sort(nodes)
    # Ensure strictly increasing
    for i in range(1, 8):
        if nodes[i] - nodes[i-1] < 1e-10:
            nodes[i] = nodes[i-1] + 1e-10
    nodes = np.clip(nodes, 1e-8, 1 - 1e-8)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
