import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0

def _time_left():
    return _DEADLINE - time.time()

def _E_cpl_exact(a, w0, wa, Om_m=0.3, Om_r=5e-5):
    """
    Compute E(a) = H(a)/H0 for exact CPL model.
    a is a numpy array.
    """
    Om_de = 1.0 - Om_m - Om_r
    term_m = Om_m * a**(-3)
    term_r = Om_r * a**(-4)
    
    # Exact CPL factor: a^(-3(1+w0+wa)) * exp(3*wa*(a-1))
    D = 1.0 + w0 + wa
    # Handle potential overflow/underflow in exp for extreme parameters
    # wa is typically small, a in [0,1], so 3*wa*(a-1) is bounded
    exp_term = np.exp(3.0 * wa * (a - 1.0))
    power_term = a**(-3.0 * D)
    term_de = Om_de * power_term * exp_term
    
    val = term_m + term_r + term_de
    val = np.maximum(val, 1e-30)
    return np.sqrt(val)

def _true_dL_fast(z, w0, wa, num_points=512):
    """
    Fast reference calculation of (1+z) * integral.
    Uses a dense fixed grid.
    """
    if z <= 0:
        return 0.0
    # Integrate in z-space directly to avoid transformation errors
    # Integral is \int_0^z dz'/E(z')
    zs = np.linspace(0, z, num_points)
    a_vals = 1.0 / (1.0 + zs)
    E_vals = _E_cpl_exact(a_vals, w0, wa)
    integrand = 1.0 / E_vals
    
    # Simpson's rule for better accuracy
    # num_points must be odd for standard Simpson's? 
    # np.trapz is simpler and with 512 points is very accurate.
    integral = np.trapz(integrand, zs)
    return (1.0 + z) * integral

def _quad_dL(z, nodes, weights, w0, wa):
    """
    Approximate dL using the 8-node quadrature rule.
    Nodes x_i in [0,1].
    Integral approx: sum w_i / E(z * x_i)
    """
    if z <= 0:
        return 0.0
    # a_i = 1 / (1 + z * x_i)
    a_vals = 1.0 / (1.0 + z * nodes)
    E_vals = _E_cpl_exact(a_vals, w0, wa)
    integral_approx = np.sum(weights / E_vals)
    return (1.0 + z) * z * integral_approx

def _decode_params(params):
    """
    Decode 17 parameters into 8 nodes and 8 weights.
    params[0:9] -> gaps t_0..t_8
    params[9:17] -> weights v_1..v_8
    """
    # Nodes from gaps
    t = params[:9]
    # Softmax for gaps
    t_max = np.max(t)
    exp_t = np.exp(t - t_max)
    gaps = exp_t / np.sum(exp_t)
    
    # Cumulative sum for nodes
    nodes = np.cumsum(gaps)
    # The last node is nodes[7]. The gap to 1 is gaps[8].
    # nodes[7] = 1 - gaps[8].
    # This ensures 0 < nodes[0] < ... < nodes[7] < 1.
    
    # Weights
    v = params[9:17]
    v_max = np.max(v)
    exp_v = np.exp(v - v_max)
    weights = exp_v / np.sum(exp_v)
    
    return nodes, weights

def _objective(params):
    if _time_left() < 1.0:
        return 1e10
        
    nodes, weights = _decode_params(params)
    
    # Test set
    # Representative CPL models
    w0_vals = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0]
    wa_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    z_vals = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    max_err = 0.0
    for w0 in w0_vals:
        for wa in wa_vals:
            for z in z_vals:
                true_val = _true_dL_fast(z, w0, wa)
                quad_val = _quad_dL(z, nodes, weights, w0, wa)
                if true_val > 1e-10:
                    rel_err = abs(quad_val - true_val) / true_val
                    if rel_err > max_err:
                        max_err = rel_err
                        
    return max_err

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 85.0
    
    # Initial guess: Uniform gaps and uniform weights
    # Uniform gaps: t_i = 0 for all i
    # Uniform weights: v_i = 0 for all i
    x0 = np.zeros(17)
    
    bounds = [(-10, 10)] * 17
    
    best_result = None
    best_score = 1e10
    
    # Phase 1: Differential Evolution
    try:
        if _time_left() > 30:
            res = differential_evolution(
                _objective, 
                bounds, 
                seed=42, 
                maxiter=100, 
                popsize=15, 
                tol=1e-8,
                polish=False
            )
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
    except Exception as e:
        pass
    
    # Phase 2: Local Refinement
    if best_result is not None and _time_left() > 10:
        try:
            res2 = minimize(
                _objective, 
                best_result, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if res2.fun < best_score:
                best_score = res2.fun
                best_result = res2.x
        except Exception as e:
            pass
            
    # Fallback to Gauss-Legendre if optimization fails
    if best_result is None:
        x_gl, w_gl = np.polynomial.legendre.leggauss(8)
        nodes = 0.5 * (x_gl + 1.0)
        weights = 0.5 * w_gl
        weights = weights / np.sum(weights)
        return {"nodes": nodes, "weights": weights}

    nodes, weights = _decode_params(best_result)
    
    # Final check
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
