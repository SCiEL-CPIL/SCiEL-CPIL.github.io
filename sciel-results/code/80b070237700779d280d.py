"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
from numpy.polynomial.legendre import leggauss
import time

def _E_of_z(z, w0, w1):
    """CPL model: E(z) = H(z)/H0, vectorized"""
    z = np.asarray(z, dtype=float)
    term = 1.0 + w0 + w1 * z / (1.0 + z)
    return (1.0 + z)**3 * term**(-3.0 * w1)

def _exact_integral(z, w0, w1):
    """Compute exact integral int_0^1 dt / E(z*t)"""
    if z < 1e-10:
        return 1.0
    f = lambda t: 1.0 / _E_of_z(np.array([z * t]), w0, w1)[0]
    result, _ = quad(f, 0, 1, limit=100, epsabs=1e-11, epsrel=1e-11)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 75.0
    
    # Cosmologies for optimization
    cosmo_list = [
        (-1.0, 0.0),
        (-1.0, 0.5),
        (-0.9, 0.3),
        (-0.8, -0.5),
        (-1.1, 0.3),
    ]
    z_list = [0.2, 0.5, 1.0, 2.0, 4.0]
    
    # Pre-compute exact integrals
    exact_cache = {}
    for (w0, w1) in cosmo_list:
        for z in z_list:
            exact_cache[(w0, w1, z)] = _exact_integral(z, w0, w1)
    
    # Fallback: Gauss-Legendre
    x_gl, w_gl = leggauss(8)
    nodes_gl = (x_gl + 1.0) / 2.0
    weights_gl = w_gl / 2.0
    
    if time.time() - start_time > 25:
        return {"nodes": nodes_gl, "weights": weights_gl}
    
    def _objective(params):
        t = params[:8]
        w = params[8:16]
        
        if np.any(np.diff(t) <= 1e-10) or t[0] <= 0 or t[7] >= 1 or np.any(w <= 0):
            return 1e6
        
        w_norm = w / np.sum(w)
        
        max_err = 0.0
        for (w0, w1) in cosmo_list:
            for z in z_list:
                exact = exact_cache[(w0, w1, z)]
                zt = z * t
                E_vals = _E_of_z(zt, w0, w1)
                quad_val = np.dot(w_norm, 1.0 / E_vals)
                rel_err = abs(quad_val - exact) / exact * 100.0
                if rel_err > max_err:
                    max_err = rel_err
        return max_err
    
    x0 = np.concatenate([nodes_gl, weights_gl])
    best_params = x0.copy()
    best_obj = _objective(x0)
    
    # Nelder-Mead
    try:
        res = minimize(_objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 200, 'fatol': 1e-5, 'adaptive': True})
        if res.fun < best_obj:
            best_obj = res.fun
            best_params = res.x
    except Exception:
        pass
    
    if time.time() > deadline - 5:
        t = best_params[:8].copy()
        w = best_params[8:16].copy()
        w = w / np.sum(w)
        for i in range(1, 8):
            if t[i] <= t[i-1]:
                t[i] = t[i-1] + 1e-8
        return {"nodes": t, "weights": w}
    
    # L-BFGS-B
    bounds = [(0.001 + i*0.0005, 0.999 - (7-i)*0.0005) for i in range(8)] + \
             [(1e-8, 1.0)] * 8
    
    try:
        res2 = minimize(_objective, best_params, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 80, 'ftol': 1e-10})
        if res2.fun < best_obj:
            best_obj = res2.fun
            best_params = res2.x
    except Exception:
        pass
    
    t = best_params[:8].copy()
    w = best_params[8:16].copy()
    w = w / np.sum(w)
    for i in range(1, 8):
        if t[i] <= t[i-1]:
            t[i] = t[i-1] + 1e-8
    
    return {"nodes": t, "weights": w}
# EVOLVE-BLOCK-END
