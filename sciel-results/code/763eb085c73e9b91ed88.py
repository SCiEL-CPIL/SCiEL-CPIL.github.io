"""Refined Cohn-Goncalves Laguerre double-root construction for Fourier sign uncertainty."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Precompute quadrature points once
_N_QUAD = 800
_X_QUAD, _W_QUAD = np.polynomial.laguerre.laggauss(_N_QUAD)

def _compute_norm_log_and_grad(roots, x_quad=None, w_quad=None):
    """
    Compute the log of the L2 norm squared of P(x) = prod (x - r_i)^2
    with weight e^{-x}, and its gradient w.r.t. roots.
    We MAXIMIZE this norm to MINIMIZE the C4 bound.
    """
    if x_quad is None:
        x_quad, w_quad = np.polynomial.laguerre.laggauss(_N_QUAD)
    m = len(roots)
    
    # log |P(x)|^2 = 4 * sum_i log|x - r_i|
    log_p_sq = np.zeros_like(x_quad)
    for r in roots:
        diff = np.abs(x_quad - r)
        diff = np.maximum(diff, 1e-300)
        log_p_sq += 4.0 * np.log(diff)
    
    log_integrand = -x_quad + log_p_sq
    max_log = np.max(log_integrand)
    if not np.isfinite(max_log):
        return -np.inf, np.zeros(m)
    
    exp_shifted = np.exp(log_integrand - max_log)
    weighted = w_quad * exp_shifted
    sum_val = np.sum(weighted)
    
    if sum_val <= 0:
        return -np.inf, np.zeros(m)
        
    log_norm_sq = max_log + np.log(sum_val)
    
    # Gradient: d/d(r_j) log_norm_sq = 4 * sum_i weighted_i * (1/(x_i - r_j))
    grad = np.zeros(m)
    for j, r in enumerate(roots):
        diff = x_quad - r
        diff = np.where(np.abs(diff) < 1e-300, 1e-300, diff)
        grad[j] = 4.0 * np.sum(weighted / diff)
    
    return log_norm_sq, grad

def _dyson_gas_init(m=12, repulsion=4.0, spread=1.0):
    """
    Initialize root positions using Dyson gas equilibrium.
    """
    roots = np.linspace(2.0 * spread, 50.0 * spread, m)
    
    for iteration in range(500):
        forces = np.zeros(m)
        for i in range(m):
            forces[i] = roots[i]
            for j in range(m):
                if i != j:
                    forces[i] -= repulsion / (roots[i] - roots[j])
        
        max_force = np.max(np.abs(forces))
        step_size = min(0.05, 1.0 / (1.0 + max_force))
        for i in range(m):
            roots[i] += step_size * forces[i]
            roots[i] = max(roots[i], 0.5)
        
        if max_force < 1e-14:
            break
    
    return np.sort(roots)

def construct():
    """
    Return 12 positive double-root locations that minimize the upper bound
    on C4 for the Cohn-Goncalves Laguerre polynomial construction (m=12, n=1).
    """
    start_time = time.time()
    deadline = start_time + 585.0
    
    seed_roots = np.array([4.0 * i for i in range(1, 13)])
    
    # Generate multiple diverse initializations
    inits = []
    try:
        inits.append(_dyson_gas_init(12, repulsion=4.0, spread=1.0))
    except:
        pass
    try:
        inits.append(_dyson_gas_init(12, repulsion=6.0, spread=1.0))
    except:
        pass
    try:
        inits.append(_dyson_gas_init(12, repulsion=4.0, spread=0.7))
    except:
        pass
    try:
        inits.append(_dyson_gas_init(12, repulsion=4.0, spread=1.3))
    except:
        pass
    inits.append(seed_roots)
    
    # Evaluate all initializations
    best_roots = seed_roots.copy()
    best_log_norm = -np.inf
    for init in inits:
        ln, _ = _compute_norm_log_and_grad(init)
        if np.isfinite(ln) and ln > best_log_norm:
            best_log_norm = ln
            best_roots = init.copy()
    
    # Objective in log-space WITHOUT sorting (to preserve gradient correctness)
    # We optimize in log-space; roots are exp(log_params). No sorting during opt.
    def _obj_and_grad(log_params):
        elapsed = time.time() - start_time
        if elapsed > deadline - 20:
            return 1e10, np.zeros(len(log_params))
        roots = np.exp(log_params)
        if np.any(roots <= 1e-8) or np.any(roots > 1e6):
            return 1e10, np.zeros(len(log_params))
        log_norm, grad = _compute_norm_log_and_grad(roots, _X_QUAD, _W_QUAD)
        if not np.isfinite(log_norm):
            return 1e10, np.zeros(len(log_params))
        # Chain rule: d(log_norm)/d(log_r_j) = r_j * d(log_norm)/d(r_j)
        grad_log = grad * roots
        return -log_norm, -grad_log
    
    def _obj(log_params):
        val, _ = _obj_and_grad(log_params)
        return val
    
    # Optimize from best initialization
    bounds = [(np.log(0.5), np.log(300.0))] * 12
    best_log_params = np.log(best_roots)
    
    try:
        # Primary optimization: L-BFGS-B
        res = minimize(_obj, best_log_params, jac=_obj_and_grad,
                      method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 300, 'maxfun': 15000, 'ftol': 1e-15, 'gtol': 1e-12})
        if res.success and res.fun < 1e9:
            cand = np.exp(res.x)
            if np.all(cand > 1e-8) and np.all(cand < 1e6):
                ln, _ = _compute_norm_log_and_grad(cand)
                if np.isfinite(ln) and ln > best_log_norm:
                    best_log_norm = ln
                    best_roots = cand.copy()
                    best_log_params = res.x.copy()
    except Exception:
        pass
    
    # Polish with tighter tolerance
    if time.time() - start_time < deadline - 60:
        try:
            res2 = minimize(_obj, best_log_params, jac=_obj_and_grad,
                          method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 500, 'maxfun': 30000, 'ftol': 1e-16, 'gtol': 1e-14})
            if res2.success and res2.fun < 1e9:
                cand = np.exp(res2.x)
                if np.all(cand > 1e-8) and np.all(cand < 1e6):
                    ln, _ = _compute_norm_log_and_grad(cand)
                    if np.isfinite(ln) and ln > best_log_norm:
                        best_log_norm = ln
                        best_roots = cand.copy()
                        best_log_params = res2.x.copy()
        except Exception:
            pass
        
        # Nelder-Mead polish
        if time.time() - start_time < deadline - 30:
            try:
                res3 = minimize(_obj, best_log_params, method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-15})
                if res3.success and res3.fun < 1e9:
                    cand = np.exp(res3.x)
                    if np.all(cand > 1e-8) and np.all(cand < 1e6):
                        ln, _ = _compute_norm_log_and_grad(cand)
                        if np.isfinite(ln) and ln > best_log_norm:
                            best_log_norm = ln
                            best_roots = cand.copy()
            except Exception:
                pass
    
    # Try other initializations if time permits
    if time.time() - start_time < deadline - 120:
        for idx, init in enumerate(inits):
            if np.array_equal(init, best_roots):
                continue
            try:
                lp = np.log(init)
                res_i = minimize(_obj, lp, jac=_obj_and_grad,
                               method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 200, 'maxfun': 10000, 'ftol': 1e-14, 'gtol': 1e-11})
                if res_i.success and res_i.fun < 1e9:
                    cand = np.exp(res_i.x)
                    if np.all(cand > 1e-8) and np.all(cand < 1e6):
                        ln, _ = _compute_norm_log_and_grad(cand)
                        if np.isfinite(ln) and ln > best_log_norm:
                            best_log_norm = ln
                            best_roots = cand.copy()
            except Exception:
                pass
            if time.time() - start_time > deadline - 60:
                break
    
    # Final: sort and validate
    best_roots = np.sort(best_roots)
    if len(best_roots) != 12 or np.any(best_roots <= 0):
        best_roots = seed_roots.copy()
    
    return best_roots.tolist()
# EVOLVE-BLOCK-END
