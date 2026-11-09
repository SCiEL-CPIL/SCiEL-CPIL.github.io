import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.integrate import quad
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin for safety

    # Define cosmological models for training
    # Expanded set for better universality
    models = [
        (0.3, 0.0, -1.0, 0.0),   # Lambda CDM
        (0.3, 0.0, -1.2, 0.0),   # Phantom
        (0.3, 0.0, -0.8, 0.0),   # Quintessence
        (0.3, 0.0, -1.0, 0.5),   # CPL
        (0.4, 0.0, -1.0, 0.0),   # High Omega_m
        (0.2, 0.0, -1.0, 0.0),   # Low Omega_m
        (0.3, 0.05, -1.0, 0.0),  # With Radiation
        (0.3, 0.0, -1.0, -0.5),  # CPL negative wa
        (0.3, 0.0, -1.5, 0.0),   # Strong Phantom
        (0.3, 0.0, -0.5, 0.0),   # Strong Quintessence
    ]
    
    # Redshifts to test - more points for better constraint
    z_vals = [0.05, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]
    
    # Pre-compute reference values
    def E_cpl(a, om_m, om_r, w0, wa):
        a = np.maximum(a, 1e-10)
        h2 = om_m * a**(-3) + om_r * a**(-4) + (1 - om_m - om_r) * a**(-3*(1 + w0 + wa))
        return np.sqrt(np.maximum(h2, 1e-30))
        
    def integrand(x, z, om_m, om_r, w0, wa):
        a = 1.0 / (1.0 + z * x)
        return 1.0 / E_cpl(a, om_m, om_r, w0, wa)
        
    ref_integrals = []
    for om_m, om_r, w0, wa in models:
        for z in z_vals:
            # High precision integration
            val, err = quad(lambda x: integrand(x, z, om_m, om_r, w0, wa), 0, 1, limit=200)
            ref_integrals.append(val)
            
    ref_integrals = np.array(ref_integrals)
    
    # Mapping from unconstrained parameters to nodes and weights
    def params_to_nodes_weights(params):
        u = params[:8]
        v = params[8:]
        
        # Nodes: cumulative sigmoid
        x = np.zeros(8)
        x[0] = 1.0 / (1.0 + np.exp(-u[0]))
        for i in range(1, 8):
            # Increment in (0, 1 - x[i-1])
            inc = (1.0 / (1.0 + np.exp(-u[i]))) * (1.0 - x[i-1])
            x[i] = x[i-1] + inc
        # Ensure strictly in (0,1)
        x = np.clip(x, 1e-9, 1.0 - 1e-9)
        
        # Weights: softmax
        w = np.exp(v - np.max(v))
        w = w / np.sum(w)
        
        return x, w
        
    def compute_errors(params):
        x, w = params_to_nodes_weights(params)
        errors = np.zeros(len(ref_integrals))
        idx = 0
        for om_m, om_r, w0, wa in models:
            for z in z_vals:
                a = 1.0 / (1.0 + z * x)
                e = E_cpl(a, om_m, om_r, w0, wa)
                approx_int = np.sum(w / e)
                
                ref_val = ref_integrals[idx]
                idx += 1
                
                # Relative residual
                errors[idx-1] = (approx_int - ref_val) / ref_val
        return errors

    def minimax_objective(params):
        """
        Smooth approximation of the minimax objective using Log-Sum-Exp.
        We want to minimize max(|err|).
        """
        errors = compute_errors(params)
        abs_errors = np.abs(errors)
        # Use a large beta to approximate max, but not too large to cause overflow/instability
        # beta=100 is usually sufficient for double precision if errors are not huge
        beta = 100.0
        # Shift for numerical stability
        m = np.max(abs_errors)
        if m > 1e-12:
            lse = m + np.log(np.sum(np.exp(beta * (abs_errors - m)))) / beta
        else:
            lse = m
        return lse

    # Initial guess: Gauss-Legendre nodes and weights on [0,1]
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    x_gl = 0.5 * x_gl + 0.5
    w_gl = 0.5 * w_gl
    
    # Invert parameterization for initial u, v
    u_init = np.zeros(8)
    u_init[0] = np.log(x_gl[0] / (1 - x_gl[0]))
    for i in range(1, 8):
        sig = (x_gl[i] - x_gl[i-1]) / (1 - x_gl[i-1])
        sig = np.clip(sig, 1e-9, 1-1e-9)
        u_init[i] = np.log(sig / (1 - sig))
        
    v_init = np.log(w_gl)
    
    x0 = np.concatenate([u_init, v_init])
    
    best_params = x0
    best_score = minimax_objective(x0)
    
    # Phase 1: Global Optimization using BFGS or L-BFGS-B on the smooth minimax surrogate
    # L-BFGS-B is good for large problems, but 16 vars is small. BFGS is often better for smooth non-convex local minima.
    try:
        # We can also try Nelder-Mead if gradients are noisy, but BFGS is faster if smooth
        res = minimize(minimax_objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    except Exception as e:
        pass
        
    # Phase 2: Local Refinement
    # Sometimes the smooth approximation has a flat region. We can do a direct search on the true max error
    # or refine the parameters with a more aggressive local method.
    
    if time.time() < deadline - 15:
        # Try a second run with a different start point or local perturbation
        # Or simply refine the current best with Nelder-Mead which doesn't rely on gradients
        def true_max_error(params):
            errors = compute_errors(params)
            return np.max(np.abs(errors))
            
        try:
            res2 = minimize(true_max_error, best_params, method='Nelder-Mead', 
                            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x
        except Exception as e:
            pass

    # Extract final nodes and weights
    nodes, weights = params_to_nodes_weights(best_params)
    
    # Ensure weights sum to 1 exactly (floating point)
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
