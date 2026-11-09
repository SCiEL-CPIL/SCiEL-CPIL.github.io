"""Chebyshev-inspired minimax quadrature design with Log-Sum-Exp objective."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad

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

def _true_dL_fast(z, w0, wa):
    # Use fixed high-order Gauss-Legendre for speed (1000x faster than quad)
    # 32-point GL is very accurate for smooth 1/E(a)
    if z == 0: return 0.0
    x_gl, w_gl = np.polynomial.legendre.leggauss(32)
    nodes_z = 0.5 * (x_gl + 1.0)
    weights_z = 0.5 * w_gl
    
    # Map to z-interval [0, z]
    # Integral is int_0^z dz' / E(a(z'))
    # Change var: u in [0,1], z' = z*u
    # dz' = z du
    # a = 1/(1+z*u)
    a_vals = 1.0 / (1.0 + z * nodes_z)
    E_vals = _E_cpl(a_vals, w0, wa)
    integral = np.sum(weights_z * (1.0 / E_vals)) * z
    return (1.0 + z) * integral

def _quad_dL(z, nodes, weights, w0, wa):
    if z == 0: return 0.0
    a_vals = 1.0 / (1.0 + z * nodes)
    E_vals = _E_cpl(a_vals, w0, wa)
    integral_approx = np.sum(weights / E_vals)
    return (1.0 + z) * z * integral_approx

def _dense_error_grid(nodes, weights):
    # Dense grid for minimax
    w0s = np.linspace(-1.5, 1.0, 15)
    was = np.linspace(-1.5, 1.0, 15)
    zs = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0])
    
    errors = []
    for w0 in w0s:
        for wa in was:
            for z in zs:
                true_val = _true_dL_fast(z, w0, wa)
                if true_val < 1e-10: continue
                quad_val = _quad_dL(z, nodes, weights, w0, wa)
                rel_err = abs(quad_val - true_val) / true_val
                errors.append(rel_err)
    return np.array(errors)

def _params_to_nodes_weights(params):
    """
    params 0-7: log-spacings for nodes (cumulative sum to ensure ordering in (0,1))
    params 8-15: log-weights
    """
    # Nodes: 
    # t_i are log-spacings. 
    # d_i = exp(t_i)
    # x_i = (d_1 + ... + d_i) / (d_1 + ... + d_8)
    # This ensures 0 < x_1 < ... < x_8 < 1
    t = params[:8]
    d = np.exp(t)
    cumsum_d = np.cumsum(d)
    total_d = cumsum_d[-1]
    nodes = cumsum_d / total_d
    
    # Weights
    u = params[8:16]
    w_raw = np.exp(u)
    weights = w_raw / np.sum(w_raw)
    
    return nodes, weights

def _objective(params):
    if _time_left() < 2.0:
        return 1e10
    
    nodes, weights = _params_to_nodes_weights(params)
    
    # Check validity (should be guaranteed by param mapping)
    if not np.all(np.diff(nodes) > 0):
        return 1e10
    if not np.all(weights > 0):
        return 1e10
        
    errors = _dense_error_grid(nodes, weights)
    if len(errors) == 0:
        return 1e10
        
    # Log-Sum-Exp surrogate for max error
    # scale = 100 for sharpness
    scale = 100.0
    max_err = np.max(errors)
    # Shift for numerical stability
    shifted = scale * (errors - max_err)
    lse = max_err + (1.0/scale) * np.log(np.sum(np.exp(shifted)))
    
    return lse

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 85.0
    
    # Initialize with Chebyshev nodes mapped to [0,1]
    # Chebyshev nodes on [-1,1]: cos((2k-1)pi / 2N)
    N = 8
    k = np.arange(1, N+1)
    x_cheb = np.cos((2*k - 1) * np.pi / (2*N))
    # Map to [0,1]
    x_cheb_01 = 0.5 * (x_cheb + 1.0)
    # Sort
    x_cheb_01 = np.sort(x_cheb_01)
    
    # Invert node mapping to get initial params t
    # x_i = cumsum(exp(t)) / sum(exp(t))
    # This is hard to invert analytically easily for arbitrary t.
    # We can just start with uniform t (which gives uniform nodes) or random.
    # Let's start with params that produce nodes close to Chebyshev.
    # Approximation: t_i = log(x_{i+1} - x_i) roughly?
    # Better: Just use random initialization and let optimizer work, or use GL.
    # Let's use GL nodes as initial guess for params.
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    x_gl_01 = np.sort(0.5 * (x_gl + 1.0))
    
    # Estimate t from x_gl_01
    # x_i = S_i / S_8
    # S_i = x_i * S_8
    # S_i = S_{i-1} + exp(t_i)
    # exp(t_i) = S_i - S_{i-1}
    # Let S_8 = 1 (scale invariance of t)
    S = x_gl_01
    S_prev = np.concatenate(([0.0], S[:-1]))
    d = S - S_prev
    d = np.maximum(d, 1e-12)
    t_init = np.log(d)
    
    # Weights: log(w_gl)
    w_gl_01 = np.sort(w_gl / np.sum(w_gl)) # Sorted to match nodes
    # Actually GL weights are symmetric, nodes sorted, weights sorted?
    # GL nodes are symmetric around 0.5. Weights are symmetric.
    # If we sort nodes, we must sort weights correspondingly.
    idx = np.argsort(x_gl)
    x_gl_01 = 0.5 * (x_gl[idx] + 1.0)
    w_gl_01 = 0.5 * w_gl[idx]
    
    # Re-estimate t for this sorted order
    S = x_gl_01
    S_prev = np.concatenate(([0.0], S[:-1]))
    d = S - S_prev
    d = np.maximum(d, 1e-12)
    t_init = np.log(d)
    
    u_init = np.log(w_gl_01)
    
    params_init = np.concatenate([t_init, u_init])
    
    best_params = params_init.copy()
    best_score = _objective(best_params)
    
    # Optimization with L-BFGS-B
    # L-BFGS-B is good for smooth objectives with bounds (though we don't have explicit bounds on params, they are unconstrained)
    
    if _time_left() > 10.0:
        try:
            res = minimize(_objective, params_init, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Refinement with SLSQP or tighter L-BFGS-B
    if _time_left() > 5.0:
        try:
            res = minimize(_objective, best_params, method='L-BFGS-B',
                          options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
            
    nodes, weights = _params_to_nodes_weights(best_params)
    
    # Final validity check
    nodes = np.clip(nodes, 1e-12, 1.0 - 1e-12)
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
