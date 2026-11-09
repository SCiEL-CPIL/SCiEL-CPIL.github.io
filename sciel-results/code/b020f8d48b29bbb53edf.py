"""Gauss-Legendre-based 8-node quadrature for luminosity distance with targeted refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.linalg import solve
from numpy.polynomial import polynomial as P

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0

def _time_left():
    return _DEADLINE - time.time()

def _E_cpl(a, w0, wa, Om_m=0.3, Om_r=5e-5):
    Om_de = 1.0 - Om_m - Om_r
    term_m = Om_m * a**(-3)
    term_r = Om_r * a**(-4)
    cpl_factor = 1.0 + wa * (1.0 - a)
    term_de = Om_de * a**(-3.0 * (1.0 + w0 + wa)) * (cpl_factor**3)
    val = term_m + term_r + term_de
    val = np.maximum(val, 1e-30)
    return np.sqrt(val)

def _true_dL_fast(z, w0, wa, n_ref=200):
    """Fast reference using high-order GL quadrature instead of adaptive quad."""
    # Use 200-point GL for very accurate reference
    x_ref, w_ref = np.polynomial.legendre.leggauss(n_ref)
    x_ref = 0.5 * (x_ref + 1.0)
    w_ref = 0.5 * w_ref
    a_vals = 1.0 / (1.0 + z * x_ref)
    E_vals = _E_cpl(a_vals, w0, wa)
    integral = np.sum(w_ref / E_vals)
    return (1.0 + z) * integral

def _quad_dL(z, nodes, weights, w0, wa):
    a_vals = 1.0 / (1.0 + z * nodes)
    E_vals = _E_cpl(a_vals, w0, wa)
    integral_approx = np.sum(weights / E_vals)
    return (1.0 + z) * z * integral_approx

def _worst_rel_error(nodes, weights, test_cosmos, test_zs):
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

def _gauss_legendre_init():
    """Get 8-point Gauss-Legendre nodes and weights on [0,1]."""
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    nodes = 0.5 * (x_gl + 1.0)
    weights = 0.5 * w_gl
    idx = np.argsort(nodes)
    nodes = nodes[idx]
    weights = weights[idx]
    weights = weights / np.sum(weights)
    return nodes, weights

def _rational_init():
    """
    Find nodes/weights via rational approximation approach.
    
    We want an 8-node quadrature that approximates ∫₀¹ f(x)dx ≈ Σ w_i f(x_i)
    where f(x) = 1/E(a(x)) with a = 1/(1+zx).
    
    Strategy: Fit 1/E(a(x)) as a rational function R(x) = (Σ_{k=0}^{m} p_k x^k) / (Σ_{k=0}^{n} q_k x^k)
    across multiple cosmologies, then derive quadrature from the rational structure.
    
    Alternative simpler approach: Use the fact that for a fixed set of nodes x_i,
    the optimal weights for a given integrand family can be found by linear least squares.
    We'll optimize nodes to minimize worst-case error over a family of integrands.
    """
    # Test cosmologies for fitting
    fit_cosmos = [
        (0.0, 0.0),    # LCDM
        (-1.0, 0.0),   # Lambda
        (0.5, -1.0),   # Quintessence
        (-1.5, 0.5),   # Phantom
        (-0.5, 0.5),   # Another CPL
        (0.3, -0.3),   # Mild CPL
        (0.0, 0.5),    # w0=0, wa=0.5
        (-0.3, -0.3),  # w0=-0.3, wa=-0.3
    ]
    fit_zs = np.array([0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0])
    
    # For each (cosmo, z), the integrand is f(x) = 1/E(1/(1+zx))
    # We want to find nodes x_i and weights w_i such that
    # Σ w_i f(x_i) ≈ ∫₀¹ f(x) dx for all f in our family
    
    # Approach: Parameterize nodes as GL nodes + small perturbations
    # Then for fixed nodes, solve for optimal weights via linear least squares
    # across all test cases.
    
    gl_nodes, gl_weights = _gauss_legendre_init()
    
    # Build design matrix: for each test case (cosmo, z), we have a target integral
    # and the values f(x_i) at our candidate nodes.
    # We want to find weights w such that Σ w_i f_j(x_i) ≈ I_j for all j
    
    n_nodes = 8
    all_targets = []
    all_F = []  # Each row: f(x_1), ..., f(x_8) for one test case
    
    for w0, wa in fit_cosmos:
        for z in fit_zs:
            target = _true_dL_fast(z, w0, wa) / ((1.0 + z) * z)  # This is ∫₀¹ f(x)dx
            # f(x) = 1/E(1/(1+zx))
            x_vals = gl_nodes  # Start with GL nodes
            a_vals = 1.0 / (1.0 + z * x_vals)
            E_vals = _E_cpl(a_vals, w0, wa)
            f_vals = 1.0 / E_vals
            all_targets.append(target)
            all_F.append(f_vals)
    
    all_targets = np.array(all_targets)
    all_F = np.array(all_F)  # shape: (n_tests, 8)
    
    # Solve least squares: minimize ||all_F @ w - all_targets||
    # This gives optimal weights for GL nodes
    w_ls, residuals, rank, sv = np.linalg.lstsq(all_F, all_targets, rcond=None)
    
    # Ensure positive weights
    w_ls = np.maximum(w_ls, 1e-10)
    w_ls = w_ls / np.sum(w_ls)
    
    # Now refine nodes using Nelder-Mead with these weights as starting point
    # But first, let's also try a rational function approach for node placement
    
    # Rational approach: fit 1/E(a(x)) ≈ R(x) = P(x)/Q(x) 
    # For each cosmo, the integrand depends on z through a = 1/(1+zx)
    # Let u = 1/(1+zx), so x = (1/u - 1)/z
    # The integral becomes ∫₀¹ 1/E(1/(1+zx)) dx
    # With substitution, this is ∫_{1/(1+z)}^{1} 1/E(u) * 1/(z(1-u)²) du ... complex
    
    # Simpler: just use the LS weights with GL nodes as initialization
    return gl_nodes, w_ls

def _params_to_nodes_weights(params):
    """
    Params 0-7: log-scale shifts from GL nodes.
    Params 8-15: log-weights.
    """
    gl_nodes, _ = _gauss_legendre_init()
    node_shifts = params[:8]
    raw_nodes = gl_nodes * np.exp(node_shifts)
    raw_nodes = np.clip(raw_nodes, 1e-10, 1.0 - 1e-10)
    for i in range(1, 8):
        if raw_nodes[i] <= raw_nodes[i-1]:
            raw_nodes[i] = raw_nodes[i-1] + 1e-10
    
    log_weights = params[8:16]
    w_raw = np.exp(log_weights)
    weights = w_raw / np.sum(w_raw)
    
    return raw_nodes, weights

def _objective(params):
    if _time_left() < 2.0:
        return 1e10
    
    nodes, weights = _params_to_nodes_weights(params)
    
    test_cosmos = [
        (0.0, 0.0),    # LCDM
        (-1.0, 0.0),   # Lambda
        (0.5, -1.0),   # Quintessence
        (-1.5, 0.5),   # Phantom
        (-0.5, 0.5),   # Another CPL
        (0.3, -0.3),   # Mild CPL
        (0.0, 0.5),    # w0=0, wa=0.5
        (-0.3, -0.3),  # w0=-0.3, wa=-0.3
    ]
    test_zs = np.array([0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0])
    
    return _worst_rel_error(nodes, weights, test_cosmos, test_zs)

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 85.0
    
    # Get rational-based initialization
    init_nodes, init_weights = _rational_init()
    
    # Convert to params
    gl_nodes, _ = _gauss_legendre_init()
    params_init = np.zeros(16)
    # Node shifts: init_nodes = gl_nodes * exp(shift) => shift = log(init_nodes/gl_nodes)
    for i in range(8):
        if init_nodes[i] > 0 and gl_nodes[i] > 0:
            params_init[i] = np.log(init_nodes[i] / gl_nodes[i])
        else:
            params_init[i] = 0.0
    params_init[8:16] = np.log(np.maximum(init_weights, 1e-15))
    
    best_params = params_init.copy()
    best_score = _objective(best_params)
    
    # Also try pure GL initialization
    params_gl = np.zeros(16)
    _, gl_w = _gauss_legendre_init()
    params_gl[8:16] = np.log(gl_w)
    score_gl = _objective(params_gl)
    if score_gl < best_score:
        best_score = score_gl
        best_params = params_gl.copy()
    
    # Nelder-Mead optimization with multiple restarts
    n_restarts = 5
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if _time_left() < 15.0:
            break
        
        if restart == 0:
            start_params = best_params.copy()
        else:
            # Perturb best params
            start_params = best_params + rng.normal(0, 0.1, 16)
            start_params = np.clip(start_params, -2, 2)
        
        try:
            res = minimize(_objective, start_params, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final refinement
    if _time_left() > 10.0:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final decode
    nodes, weights = _params_to_nodes_weights(best_params)
    
    # Ensure validity
    nodes = np.clip(nodes, 1e-12, 1.0 - 1e-12)
    # Ensure strictly increasing
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-12
    weights = np.maximum(weights, 1e-15)
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
