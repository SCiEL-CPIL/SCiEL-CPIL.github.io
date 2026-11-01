"""Spectral moment approach: minimize uncertainty bound via exact Gauss-Hermite quadrature
and global-to-local optimization strategy."""

import numpy as np
from scipy.optimize import minimize, dual_annealing, minimize_scalar
from scipy.special import roots_hermite
import time

_DEADLINE = time.time() + 160

# Pre-compute Gauss-Hermite nodes and weights for exact integration
# We need exactness up to degree 28 (from x^4 * |P(x)^2| where P is deg 12)
# 2*N - 1 >= 28 => N >= 14.5 => N=15
# Use N=20 for extra safety and numerical stability
_N_QUAD = 20
_x_nodes, _w_weights = roots_hermite(_N_QUAD)

def _make_g_coefficients(a0, a4, a8):
    """
    Given free coefficients a0, a4, a8, compute all polynomial coefficients
    P(x) = sum_{k=0}^{6} p_k x^{2k} for g(x) = P(x) e^{-x^2/2}.
    a12 is fixed to enforce g(0) = 0.
    """
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    p0 = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    p1 = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p2 = a4 + 210.0 * a8 + 13860.0 * a12
    p3 = -28.0 * a8 - 6930.0 * a12
    p4 = a8 + 1155.0 * a12
    p5 = -66.0 * a12
    p6 = a12
    return np.array([p0, p1, p2, p3, p4, p5, p6]), a12

def _compute_moments(p, nodes, weights):
    """
    Compute m_k = integral x^{2k} |P(x)|^2 e^{-x^2} dx for k=0,1,2
    using Gauss-Hermite quadrature (exact for polynomials).
    """
    x2 = nodes**2
    P_vals = np.zeros_like(nodes)
    for i in range(7):
        P_vals += p[i] * (x2 ** i)
    
    P_sq = P_vals * P_vals
    
    m = np.zeros(3)
    for k in range(3):
        m[k] = np.sum(weights * (nodes ** (2*k)) * P_sq)
        
    return m

def _compute_fourier_moments(p, nodes, weights):
    """
    Compute n_k = 2*pi * integral xi^{2k} |Q(xi)|^2 e^{-xi^2} dxi for k=0,1,2
    where Q(xi) = sum p_k (-1)^k xi^{2k}
    """
    q = p * ((-1.0) ** np.arange(7))
    
    x2 = nodes**2
    Q_vals = np.zeros_like(nodes)
    for i in range(7):
        Q_vals += q[i] * (x2 ** i)
        
    Q_sq = Q_vals * Q_vals
    
    n = np.zeros(3)
    for k in range(3):
        n[k] = 2.0 * np.pi * np.sum(weights * (nodes ** (2*k)) * Q_sq)
        
    return n

def _bound_value(params):
    """Compute the uncertainty bound for given parameters."""
    p, _ = _make_g_coefficients(params[0], params[1], params[2])
    
    m = _compute_moments(p, _x_nodes, _w_weights)
    n = _compute_fourier_moments(p, _x_nodes, _w_weights)
    
    if m[0] <= 0 or n[0] <= 0 or not np.all(np.isfinite(m)) or not np.all(np.isfinite(n)):
        return 1e18
    
    vf = m[1] / m[0]
    vF = n[1] / n[0]
    
    if vf <= 0 or vF <= 0:
        return 1e18
        
    return np.sqrt(vf * vF)

def _bound_value_log(params):
    """Log of the uncertainty bound for better conditioning."""
    v = _bound_value(params)
    if v >= 1e18:
        return 1e18
    return np.log(v)

def construct():
    """
    Find optimal Hermite coefficients to minimize the uncertainty bound.
    Returns array of three coefficients [a0, a4, a8].
    """
    # Seed starting point (known good local minimum)
    seed = np.array([-1.13, 0.04, 0.000308642])
    best_val = _bound_value(seed)
    best_params = seed.copy()
    
    # Stage 1: Global Search with dual_annealing
    # Use tighter bounds based on expected scale
    bounds = [(-3.0, 3.0), (-2.0, 2.0), (-0.5, 0.5)]
    
    if time.time() < _DEADLINE - 80:
        try:
            res_global = dual_annealing(
                _bound_value, 
                bounds, 
                seed=42, 
                maxiter=200, 
                initial_temp=1000.0, 
                restart_temp_ratio=1e-5, 
                visit=2.0, 
                maxiter_no_improvement=15,
                polish=True,
                x0=seed
            )
            if res_global.fun < best_val and np.isfinite(res_global.fun):
                best_val = res_global.fun
                best_params = res_global.x.copy()
        except Exception:
            pass

    # Stage 2: Multi-start local optimization
    if time.time() < _DEADLINE - 30:
        # Try several starting points around the best found
        starts = [best_params.copy()]
        # Add perturbations
        for scale in [0.1, 0.05, 0.01]:
            for dx in [-scale, 0, scale]:
                for dy in [-scale, 0, scale]:
                    for dz in [-scale*0.1, 0, scale*0.1]:
                        candidate = best_params + np.array([dx, dy, dz])
                        if np.all(np.abs(candidate) < [5, 5, 1]):
                            starts.append(candidate)
        
        for start in starts[:20]:  # Limit number of starts
            if time.time() > _DEADLINE - 20:
                break
            try:
                res_local = minimize(
                    _bound_value, 
                    start, 
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13, 'maxfun': 10000}
                )
                if res_local.fun < best_val and np.isfinite(res_local.fun):
                    best_val = res_local.fun
                    best_params = res_local.x.copy()
            except Exception:
                pass

    # Stage 3: High-precision Nelder-Mead polish
    if time.time() < _DEADLINE - 5:
        try:
            res_nm = minimize(
                _bound_value, 
                best_params, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res_nm.fun < best_val and np.isfinite(res_nm.fun):
                best_val = res_nm.fun
                best_params = res_nm.x.copy()
        except Exception:
            pass

    # Stage 4: Final L-BFGS-B with very tight tolerances
    if time.time() < _DEADLINE - 1:
        try:
            res_final = minimize(
                _bound_value, 
                best_params, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-15, 'maxfun': 50000}
            )
            if res_final.fun < best_val and np.isfinite(res_final.fun):
                best_val = res_final.fun
                best_params = res_final.x.copy()
        except Exception:
            pass

    if best_params is None or not np.all(np.isfinite(best_params)):
        return seed
        
    return np.array(best_params)
