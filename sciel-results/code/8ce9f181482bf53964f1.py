"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import quad
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

def _E_cpl(a, w0, wa):
    """H(a)/H0 for CPL model: E(z) = (1+z)^(3(1+w0+wa)) * exp(-3*wa*z)"""
    z = 1.0 / a - 1.0
    return (1.0 + z) ** (3.0 * (1.0 + w0 + wa)) * np.exp(-3.0 * wa * z)

def _integrand(z, w0, wa):
    """z/(1+z) * 1/E(a) = z/(1+z) * 1/E(z)"""
    if z == 0.0:
        return 0.0
    a = 1.0 / (1.0 + z)
    Ez = (1.0 + z) ** (3.0 * (1.0 + w0 + wa)) * np.exp(-3.0 * wa * z)
    return z / (1.0 + z) / Ez

def _exact_integral(w0, wa, zmax=10.0):
    """Exact integral of z/(1+z)/E(a) dz from 0 to zmax"""
    result, _ = quad(_integrand, 0, zmax, args=(w0, wa), limit=200, epsabs=1e-12, epsrel=1e-12)
    return result

def _quadrature_value(nodes, weights, w0, wa, zmax=10.0):
    """Compute quadrature approximation of the integral"""
    z_vals = zmax * nodes
    total = 0.0
    for i in range(len(nodes)):
        z = z_vals[i]
        if z == 0.0:
            total += weights[i] * 0.0
        else:
            a = 1.0 / (1.0 + z)
            Ez = (1.0 + z) ** (3.0 * (1.0 + w0 + wa)) * np.exp(-3.0 * wa * z)
            total += weights[i] * z / (1.0 + z) / Ez
    return total

def _objective(params, models, exact_vals, zmax=10.0):
    """Worst-case relative error (in percent) across models"""
    n = 8
    log_nodes = params[:n]
    log_weights = params[n:]
    
    # Convert to nodes in (0,1) strictly increasing
    # Use softplus or tanh to ensure (0,1)
    # nodes = sigmoid(log_nodes) but need strictly increasing
    # Better: use cumulative approach
    # Actually, let's use a different parameterization:
    # nodes are in (0,1), strictly increasing
    # Use: node_i = (i-1 + exp(raw_i)) / (n + sum(exp(raw_j))) ... too complex
    
    # Simpler: nodes = sigmoid(log_nodes), then sort and ensure strict increase
    # But sorting breaks the mapping. Let's use:
    # raw -> strictly increasing in (0,1) via: 
    #   Let s_i = softmax(raw)[:n] normalized, then cumsum gives increasing in (0,1)
    # Actually simplest: just use the raw as log-spaced or direct with constraints
    
    # Direct approach: nodes = sigmoid(log_nodes), enforce increasing by construction
    # Use: node_i = (i + sigmoid(log_nodes_i)) / (n + 1)  -- this ensures strictly increasing in (0,1)
    nodes = (np.arange(n) + 1.0 + 1.0 / (1.0 + np.exp(-log_nodes))) / (n + 1.0)
    # This gives nodes in (i/(n+1), (i+2)/(n+1)) subset, strictly increasing
    # Actually: node_i is in ((i+1)/(n+1), (i+2)/(n+1)) for i=0..n-1
    # So node_0 in (1/9, 2/9), node_7 in (8/9, 1) -- good, all in (0,1)
    
    weights = np.exp(log_weights)
    weights = weights / weights.sum()
    
    worst_err = 0.0
    for j in range(len(models)):
        w0, wa = models[j]
        exact = exact_vals[j]
        if exact < 1e-15:
            continue
        approx = _quadrature_value(nodes, weights, w0, wa, zmax)
        rel_err = abs(approx - exact) / abs(exact) * 100.0
        worst_err = max(worst_err, rel_err)
    return worst_err

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    # Define representative CPL models to optimize against
    # Cover a range of w0 and wa values
    models = [
        (-1.0, 0.0),    # LambdaCDM
        (-1.0, 0.5),    # w0=-1, wa=0.5
        (-1.0, -0.5),   # w0=-1, wa=-0.5
        (-0.5, 0.0),    # w0=-0.5, wa=0
        (-0.5, 0.5),
        (-0.5, -0.5),
        (-1.5, 0.5),
        (-1.5, -0.5),
        (-0.9, 0.3),
        (-1.1, 0.2),
        (-0.7, 0.1),
        (-1.3, -0.3),
        (-1.0, 1.0),
        (-1.0, -1.0),
        (-0.8, 0.4),
        (-1.2, 0.6),
    ]
    
    zmax = 10.0
    
    # Compute exact integrals
    exact_vals = []
    for w0, wa in models:
        if time.time() > deadline:
            break
        exact_vals.append(_exact_integral(w0, wa, zmax))
    
    # If we couldn't compute all, use what we have
    n_models = len(exact_vals)
    models_used = models[:n_models]
    
    # Start from Gauss-Legendre as initial guess
    from numpy.polynomial.legendre import leggauss
    x_gl, w_gl = leggauss(8)
    nodes_gl = (x_gl + 1.0) / 2.0
    weights_gl = w_gl / 2.0
    
    # Convert to our parameterization: node_i = (i + 1 + sigmoid(log_node_i)) / 9
    # So sigmoid(log_node_i) = node_i * 9 - (i + 1)
    # log_node_i = logit(node_i * 9 - (i + 1))
    log_nodes_init = np.zeros(8)
    for i in range(8):
        val = nodes_gl[i] * 9.0 - (i + 1.0)
        # val should be in (0,1) for sigmoid
        val = np.clip(val, 1e-6, 1.0 - 1e-6)
        log_nodes_init[i] = np.log(val / (1.0 - val))
    
    log_weights_init = np.log(weights_gl)
    
    params_init = np.concatenate([log_nodes_init, log_weights_init])
    
    # Try Nelder-Mead optimization
    best_params = params_init.copy()
    best_err = _objective(best_params, models_used, exact_vals, zmax)
    
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                _objective, params_init,
                args=(models_used, exact_vals, zmax),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            if result.fun < best_err:
                best_params = result.x
                best_err = result.fun
        except Exception:
            pass
    
    # Try a few random restarts if time permits
    rng = np.random.default_rng(42)
    for restart in range(5):
        if time.time() > deadline - 5.0:
            break
        params_try = params_init + rng.normal(0, 0.5, size=16)
        try:
            result = minimize(
                _objective, params_try,
                args=(models_used, exact_vals, zmax),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            if result.fun < best_err:
                best_params = result.x
                best_err = result.fun
        except Exception:
            pass
    
    # Extract final nodes and weights
    n = 8
    log_nodes = best_params[:n]
    log_weights = best_params[n:]
    
    nodes = (np.arange(n) + 1.0 + 1.0 / (1.0 + np.exp(-log_nodes))) / (n + 1.0)
    weights = np.exp(log_weights)
    weights = weights / weights.sum()
    
    # Verify constraints
    # Nodes strictly increasing in (0,1)
    if not np.all(np.diff(nodes) > 0) or nodes[0] <= 0 or nodes[-1] >= 1:
        # Fall back to Gauss-Legendre
        nodes = nodes_gl
        weights = weights_gl
    
    # Ensure weights positive and sum to 1
    if not np.all(weights > 0):
        weights = weights_gl
        nodes = nodes_gl
    weights = weights / weights.sum()
    
    # Final check: sum within 1e-12
    if abs(weights.sum() - 1.0) > 1e-12:
        weights = weights / weights.sum()
    
    if time.time() > deadline:
        # Emergency: return Gauss-Legendre
        return {"nodes": nodes_gl.tolist(), "weights": weights_gl.tolist()}
    
    return {"nodes": nodes.tolist(), "weights": weights.tolist()}
# EVOLVE-BLOCK-END
