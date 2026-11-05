"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, linprog

_START = time.time()
_DEADLINE = 80.0  # safety margin before 90s kill

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _E_of_x(x, w0, wa):
    """E(a) in terms of x=z/(1+z); a=1-x."""
    # E(a) = a^{-3(1+w0+wa)} * exp(3*wa*(1-a)) = (1-x)^{-3(1+w0+wa)} * exp(3*wa*x)
    return (1.0 - x) ** (-3.0 * (1.0 + w0 + wa)) * np.exp(3.0 * wa * x)

def _integrand(x, w0, wa):
    """f(x) = x^2 / (1-x) / E(a)."""
    return x * x / (1.0 - x) / _E_of_x(x, w0, wa)

def _true_integral(w0, wa):
    """High-accuracy reference integral via Gauss-Legendre (200 nodes)."""
    nodes, weights = np.polynomial.legendre.leggauss(200)
    x = 0.5 * (nodes + 1.0)
    vals = _integrand(x, w0, wa) * 0.5
    return np.sum(weights * vals)

def _quad_eval(nodes, weights, w0, wa):
    return np.sum(weights * _integrand(nodes, w0, wa))

def _build_design_set():
    """Representative CPL models spanning plausible parameter space."""
    w0s = [-1.5, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.5]
    was = [-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0]
    models = []
    for w0 in w0s:
        for wa in was:
            models.append((w0, wa))
    # Add a few edge cases
    models.extend([(-1.6, 1.5), (-0.7, -1.2), (-1.0, 2.0), (-1.4, -0.8)])
    return models

def _log_barrier_objective(params, nodes0, weights0, design_models):
    """Minimize max relative error with log-barrier for positivity/sum constraints."""
    n = 8
    nodes = params[:n]
    log_w = params[n:]
    weights = np.exp(log_w)

    # Soft constraints via large penalties
    penalty = 0.0
    # Positivity of weights is automatic (exp)
    # Nodes in (0,1) and strictly increasing
    if nodes[0] <= 1e-10 or nodes[-1] >= 1.0 - 1e-10:
        penalty += 1e6
    diffs = np.diff(nodes)
    if np.any(diffs <= 1e-10):
        penalty += 1e6 * np.sum(diffs[diffs <= 1e-10])
    # Weights sum to 1
    penalty += 1e4 * (np.sum(weights) - 1.0) ** 2

    if penalty > 1e5:
        return penalty

    max_rel = 0.0
    for (w0, wa) in design_models:
        true_val = _true_integral(w0, wa)
        quad_val = _quad_eval(nodes, weights, w0, wa)
        rel = abs(quad_val - true_val) / abs(true_val)
        if rel > max_rel:
            max_rel = rel
    return max_rel + penalty

def _nelder_mead(params0, design_models, max_iter=3000):
    """Nelder-Mead with time check."""
    def obj(p):
        if _remaining() < 2.0:
            return 1e10
        return _log_barrier_objective(p, None, None, design_models)

    res = minimize(obj, params0, method='Nelder-Mead',
                   options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12,
                            'adaptive': True})
    return res.x

def _project_to_valid(params):
    """Project raw params to valid nodes/weights."""
    n = 8
    nodes = params[:n]
    log_w = params[n:]
    weights = np.exp(log_w)
    weights = weights / np.sum(weights)
    # Ensure nodes strictly increasing in (0,1)
    nodes = np.sort(nodes)
    # Enforce minimum spacing
    eps = 1e-8
    for i in range(1, n):
        if nodes[i] - nodes[i-1] < eps:
            nodes[i] = nodes[i-1] + eps
    nodes[0] = max(nodes[0], eps)
    nodes[-1] = min(nodes[-1], 1.0 - eps)
    return nodes, weights

def construct():
    design_models = _build_design_set()
    n = 8

    # Gauss-Legendre initialization (known to avoid local minima)
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n)
    nodes0 = 0.5 * (gl_nodes + 1.0)  # map [-1,1] -> [0,1]
    weights0 = gl_weights / 2.0
    weights0 = weights0 / np.sum(weights0)

    params0 = np.concatenate([nodes0, np.log(weights0)])

    best_params = params0.copy()
    best_score = _log_barrier_objective(params0, None, None, design_models)

    # Nelder-Mead optimization
    if _remaining() > 5.0:
        try:
            res_params = _nelder_mead(params0, design_models, max_iter=2000)
            score = _log_barrier_objective(res_params, None, None, design_models)
            if score < best_score:
                best_params = res_params
                best_score = score
        except Exception:
            pass

    # Local refinement with tighter tolerance if time permits
    if _remaining() > 3.0:
        try:
            res_params2 = _nelder_mead(best_params, design_models, max_iter=1000)
            score2 = _log_barrier_objective(res_params2, None, None, design_models)
            if score2 < best_score:
                best_params = res_params2
                best_score = score2
        except Exception:
            pass

    nodes, weights = _project_to_valid(best_params)

    # Final sanity: ensure weights sum to 1 and are positive
    weights = np.maximum(weights, 1e-15)
    weights = weights / np.sum(weights)
    nodes = np.clip(nodes, 1e-12, 1.0 - 1e-12)
    # Re-enforce strict increase
    for i in range(1, n):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-10

    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
