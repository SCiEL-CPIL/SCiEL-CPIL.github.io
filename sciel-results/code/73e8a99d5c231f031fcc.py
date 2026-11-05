"""Cosmology: universal luminosity-distance quadrature.
Strategy: Smooth epigraph + Nelder-Mead on 16-D space.
Uses log-sum-exp soft-max surrogate for the worst-case relative error
across a diverse CPL model library, then polishes with exact max.
"""
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

START = time.time()
DEADLINE = 85.0  # safety margin before 90s kill

def _fallback():
    n = 8
    return {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}

def _E_cpl(a, w0, wa):
    """CPL E(a) for flat universe: E(a) = (Om_m a^-3 + Om_l a^(-3(1+w)))^0.5
    With Om_m=0.3, Om_l=0.7, w = w0 + wa*(1-a)."""
    Om_m = 0.3
    Om_l = 0.7
    w = w0 + wa * (1.0 - a)
    # E^2 = Om_m * a^-3 + Om_l * a^(-3*(1+w))
    E2 = Om_m * a**(-3.0) + Om_l * a**(-3.0 * (1.0 + w))
    return np.sqrt(np.maximum(E2, 1e-30))

def _E_wcdm(a, w):
    """wCDM: constant w."""
    Om_m = 0.3
    Om_l = 0.7
    E2 = Om_m * a**(-3.0) + Om_l * a**(-3.0 * (1.0 + w))
    return np.sqrt(np.maximum(E2, 1e-30))

def _build_library():
    """Diverse CPL model family for min-max training."""
    lib = []
    zs = np.array([0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0])
    
    # Broad CPL grid
    for w0 in np.arange(-1.5, 0.51, 0.25):
        for wa in np.arange(-1.5, 1.51, 0.25):
            lib.append((zs, _E_cpl, (w0, wa)))
    
    # Additional wCDM points
    for w in np.arange(-1.5, 0.51, 0.25):
        lib.append((zs, _E_wcdm, (w,)))
    
    return lib

_LIB = None

def _relerr_all(nodes, weights, z_list, Efun, params):
    """Compute relative errors for all z values for one model."""
    g = np.polynomial.legendre.leggauss(80)
    xi, wi = g
    
    errs = np.empty(len(z_list))
    for idx, z in enumerate(z_list):
        # Reference: Gauss-Legendre
        a_ref = 1.0 / (1.0 + z * 0.5 * (xi + 1.0))
        if hasattr(Efun, '__name__') and Efun.__name__ == '_E_cpl':
            E_ref = Efun(a_ref, params[0], params[1])
            E_quad = Efun(1.0 / (1.0 + z * nodes), params[0], params[1])
        else:
            E_ref = Efun(a_ref, params[0])
            E_quad = Efun(1.0 / (1.0 + z * nodes), params[0])
        
        ref = z * 0.5 * np.sum(wi / E_ref)
        quad = z * np.sum(weights / E_quad)
        
        if ref < 1e-15:
            errs[idx] = 0.0
        else:
            errs[idx] = abs(quad - ref) / ref
    return errs

def _softmax_error(nodes, weights, temperature=0.01):
    """Log-sum-exp soft-max of relative errors across all models and z values."""
    all_errs = []
    for zs, Efun, params in _LIB:
        errs = _relerr_all(nodes, weights, zs, Efun, params)
        all_errs.extend(errs.tolist())
    
    all_errs = np.array(all_errs)
    if len(all_errs) == 0:
        return 0.0
    
    # Soft-max via log-sum-exp for numerical stability
    max_e = np.max(all_errs)
    if max_e < 1e-15:
        return max_e
    # Soft approximation to max: (1/beta) * log(sum(exp(beta*err)))
    beta = 1.0 / temperature
    shifted = beta * (all_errs - max_e)
    logsumexp = np.log(np.sum(np.exp(shifted)))
    soft_max = max_e + np.log(np.sum(np.exp(shifted))) / beta
    return soft_max

def _exact_max_error(nodes, weights):
    """Exact worst-case relative error."""
    max_err = 0.0
    for zs, Efun, params in _LIB:
        errs = _relerr_all(nodes, weights, zs, Efun, params)
        e = np.max(errs)
        if e > max_err:
            max_err = e
    return max_err

def _unconstrained_objective(x, temperature=0.01):
    """Objective on unconstrained parameterization.
    x = [node_offsets(8), log_weights(8)]
    Nodes parametrized as cumulative positive increments to ensure ordering.
    Weights parametrized as log-space, normalized.
    """
    node_params = x[:8]
    log_w = x[8:16]
    
    # Convert node params to actual nodes: cumulative sum of exp ensures positive increments
    # Start from a base, add positive increments
    increments = np.exp(node_params)
    # Normalize to fit in [0, 1]
    total = np.sum(increments)
    nodes = np.cumsum(increments) / total
    # Ensure strictly in (0,1)
    nodes = np.clip(nodes, 1e-8, 1.0 - 1e-8)
    # Enforce strict ordering
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-8
    nodes[7] = min(nodes[7], 1.0 - 1e-8)
    
    # Weights
    w = np.exp(log_w)
    w = w / np.sum(w)
    
    return _softmax_error(nodes, w, temperature)

def _polish(nodes, weights, max_iter=50):
    """High-precision polish using exact max with coordinate descent."""
    best_err = _exact_max_error(nodes, weights)
    
    for iteration in range(max_iter):
        if time.time() > DEADLINE - 3.0:
            break
        
        improved = False
        
        # Optimize each node
        for i in range(8):
            if time.time() > DEADLINE - 3.0:
                break
            if i == 0:
                lb = 1e-8
                ub = nodes[1] - 1e-8
            elif i == 7:
                lb = nodes[6] + 1e-8
                ub = 1.0 - 1e-8
            else:
                lb = nodes[i-1] + 1e-8
                ub = nodes[i+1] - 1e-8
            
            if lb >= ub:
                continue
            
            def obj_node(val):
                nodes[i] = val
                return _exact_max_error(nodes, weights)
            
            res = minimize_scalar(obj_node, bounds=(lb, ub), method='bounded',
                                  options={'xatol': 1e-10})
            if res.success and res.fun < best_err:
                nodes[i] = res.x
                best_err = res.fun
                improved = True
        
        # Optimize weights (fix sum=1 by adjusting one)
        for i in range(8):
            if time.time() > DEADLINE - 3.0:
                break
            
            # Parametrize: vary log_w[i], renormalize
            log_w = np.log(weights + 1e-30)
            
            def obj_logw(val):
                log_w[i] = val
                w = np.exp(log_w)
                w = w / np.sum(w)
                return _exact_max_error(nodes, w)
            
            res = minimize_scalar(obj_logw, bounds=(-12, 12), method='bounded',
                                  options={'xatol': 1e-10})
            if res.success:
                log_w[i] = res.x
                w_new = np.exp(log_w)
                w_new = w_new / np.sum(w_new)
                e_new = _exact_max_error(nodes, w_new)
                if e_new < best_err:
                    weights[:] = w_new
                    best_err = e_new
                    improved = True
        
        if not improved:
            break
    
    return nodes, weights, best_err

def construct():
    global _LIB
    _LIB = _build_library()
    
    fallback = _fallback()
    
    # Initial parameterization
    init_nodes = (np.arange(8) + 0.5) / 8.0
    # Convert to cumulative increments
    increments = np.diff(np.concatenate([[0], init_nodes, [1]]))
    # Remove first and last (which are init_nodes[0] and 1-init_nodes[7])
    node_params = np.log(increments[1:8] + 1e-15)  # 7 increments between 8 nodes
    # Actually we need 8 params for 8 nodes via cumulative
    # Let's use: nodes[i] = cumsum(exp(p[0:i+1])) / sum(exp(p))
    # So p has 8 elements
    # For uniform: increments all equal
    p_init = np.zeros(8)
    log_w_init = np.zeros(8)
    x0 = np.concatenate([p_init, log_w_init])
    
    # Nelder-Mead on smooth surrogate
    best_nodes = fallback["nodes"].copy()
    best_weights = fallback["weights"].copy()
    best_err = _exact_max_error(best_nodes, best_weights)
    
    # Try multiple temperatures for soft-max
    for temp in [0.05, 0.02, 0.01, 0.005]:
        if time.time() > DEADLINE - 15.0:
            break
        
        def obj_temp(x, t=temp):
            return _unconstrained_objective(x, temperature=t)
        
        try:
            res = minimize(obj_temp, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8,
                                   'adaptive': True})
            
            if res.success or res.fun < best_err:
                # Extract nodes and weights
                node_params = res.x[:8]
                log_w = res.x[8:16]
                
                increments = np.exp(node_params)
                total = np.sum(increments)
                nodes = np.cumsum(increments) / total
                nodes = np.clip(nodes, 1e-8, 1.0 - 1e-8)
                for i in range(1, 8):
                    if nodes[i] <= nodes[i-1]:
                        nodes[i] = nodes[i-1] + 1e-8
                nodes[7] = min(nodes[7], 1.0 - 1e-8)
                
                w = np.exp(log_w)
                w = w / np.sum(w)
                
                e = _exact_max_error(nodes, w)
                if e < best_err:
                    best_err = e
                    best_nodes = nodes.copy()
                    best_weights = w.copy()
        except Exception:
            pass
    
    # Polish with exact max
    if time.time() < DEADLINE - 5.0:
        best_nodes, best_weights, best_err = _polish(best_nodes, best_weights, max_iter=30)
    
    # Final validation
    if not np.all(np.diff(best_nodes) > 0):
        return fallback
    if np.any(best_weights <= 0):
        return fallback
    
    best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}
