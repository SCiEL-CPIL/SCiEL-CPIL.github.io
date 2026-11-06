import numpy as np
import time
import scipy.optimize as so
import scipy.integrate as si

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    Omega_m = 0.3
    Omega_L = 0.7

    # Generate diverse test cases
    # w0: -2 to 0.5
    # wa: -2 to 2
    # z: 0.05 to 10 (log spaced)
    w0_vals = np.linspace(-2.0, 0.5, 20)
    wa_vals = np.linspace(-2.0, 2.0, 20)
    z_vals = np.logspace(np.log10(0.05), np.log10(10.0), 10)
    
    test_cases = []
    for w0 in w0_vals:
        for wa in wa_vals:
            for z in z_vals:
                test_cases.append((z, w0, wa))
                
    test_cases = np.array(test_cases)
    N_cases = len(test_cases)
    
    # Compute exact integrals
    exact_vals = np.zeros(N_cases)
    
    def _compute_exact_single(z, w0, wa):
        def integrand(t):
            a = 1.0 / (1.0 + z * t)
            # Standard CPL form
            # Omega_L(a) = Omega_L0 * a^(-3(1+w0+wa)) * (1-a)^(3wa)
            term_m = Omega_m * (a ** -3)
            
            # Handle numerical issues
            if a <= 0 or a >= 1:
                return 0.0
            
            log_term_L = -3.0 * (1.0 + w0 + wa) * np.log(a) + 3.0 * wa * np.log(1.0 - a)
            # Clip to avoid overflow/underflow
            log_term_L = np.clip(log_term_L, -700, 700)
            term_L = Omega_L * np.exp(log_term_L)
            
            E2 = term_m + term_L
            if E2 <= 0:
                return 0.0
            E = np.sqrt(E2)
            return 1.0 / E
        
        res, err = si.quad(integrand, 0.0, 1.0, limit=100, epsabs=1e-10, epsrel=1e-10)
        return res

    for i in range(N_cases):
        if time.time() > deadline - 30:
            break
        z, w0, wa = test_cases[i]
        try:
            exact_vals[i] = _compute_exact_single(z, w0, wa)
        except Exception:
            exact_vals[i] = 1.0

    # Filter valid cases
    valid_mask = exact_vals > 1e-6
    if np.sum(valid_mask) < 10:
        # Fallback to Gauss-Legendre
        nodes, weights = np.polynomial.legendre.leggauss(8)
        nodes = (nodes + 1.0) / 2.0
        weights = weights / 2.0
        return {"nodes": nodes, "weights": weights}
        
    test_cases = test_cases[valid_mask]
    exact_vals = exact_vals[valid_mask]
    N_cases = len(test_cases)
    
    z_arr = test_cases[:, 0]
    w0_arr = test_cases[:, 1]
    wa_arr = test_cases[:, 2]

    def _quadrature_vec(nodes, weights):
        # nodes: (8,), weights: (8,)
        # z_arr: (N,)
        # Compute a_i for all cases
        # A: (N, 8)
        A = 1.0 / (1.0 + np.outer(z_arr, nodes))
        A = np.clip(A, 1e-12, 1.0 - 1e-12)
        
        # Compute E
        term_m = Omega_m * (A ** -3)
        
        log_term_L = -3.0 * (1.0 + w0_arr[:, None] + wa_arr[:, None]) * np.log(A) + 3.0 * wa_arr[:, None] * np.log(1.0 - A)
        log_term_L = np.clip(log_term_L, -700, 700)
        term_L = Omega_L * np.exp(log_term_L)
        
        E2 = term_m + term_L
        E2 = np.maximum(E2, 1e-300)
        E = np.sqrt(E2)
        inv_E = 1.0 / E
        
        # Quadrature sum: sum_i w_i * inv_E[:, i]
        quad_vals = inv_E @ weights
        
        return quad_vals

    def _max_rel_error(params):
        # params: [x1, y1..y7, logw1..logw8]
        x1 = params[0]
        y = params[1:8]
        logw = params[8:16]
        
        x = np.zeros(8)
        x[0] = x1
        for i in range(1, 8):
            x[i] = x[i-1] + np.exp(y[i-1])
            
        nodes = 1.0 / (1.0 + np.exp(-x))
        nodes = np.clip(nodes, 1e-12, 1.0 - 1e-12)
        
        w = np.exp(logw)
        w = w / np.sum(w)
        
        quad_vals = _quadrature_vec(nodes, w)
        rel_errs = np.abs(quad_vals - exact_vals) / exact_vals
        max_err = np.max(rel_errs)
        
        # Add penalty if nodes are not strictly increasing (should be handled by param)
        # But check for numerical issues
        if np.any(np.diff(nodes) <= 0):
            return 1e6
        if np.any(w <= 0):
            return 1e6
            
        return max_err

    # Initial guess: Gauss-Legendre
    nodes_init, weights_init = np.polynomial.legendre.leggauss(8)
    nodes_init = (nodes_init + 1.0) / 2.0
    weights_init = weights_init / 2.0
    
    x_init = np.log(nodes_init / (1.0 - nodes_init))
    params_init = np.zeros(16)
    params_init[0] = x_init[0]
    for i in range(1, 8):
        gap = x_init[i] - x_init[i-1]
        if gap <= 0:
            gap = 1e-6
        params_init[1+i-1] = np.log(gap)
    params_init[8:16] = np.log(weights_init)
    
    best_params = params_init
    best_err = _max_rel_error(params_init)
    
    # Optimization
    # Use Nelder-Mead for minimax
    if time.time() < deadline - 10:
        try:
            res = so.minimize(_max_rel_error, params_init, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_err:
                best_err = res.fun
                best_params = res.x
        except Exception:
            pass

    # Second pass with Powell if time remains
    if time.time() < deadline - 5:
        try:
            res2 = so.minimize(_max_rel_error, best_params, method='Powell',
                               options={'maxiter': 500, 'xtol': 1e-12, 'ftol': 1e-14})
            if res2.fun < best_err:
                best_err = res2.fun
                best_params = res2.x
        except Exception:
            pass

    # Extract final nodes and weights
    x1 = best_params[0]
    y = best_params[1:8]
    logw = best_params[8:16]
    
    x = np.zeros(8)
    x[0] = x1
    for i in range(1, 8):
        x[i] = x[i-1] + np.exp(y[i-1])
        
    final_nodes = 1.0 / (1.0 + np.exp(-x))
    final_nodes = np.clip(final_nodes, 1e-12, 1.0 - 1e-12)
    
    final_weights = np.exp(logw)
    final_weights = final_weights / np.sum(final_weights)
    
    # Final cleanup
    for i in range(1, 8):
        if final_nodes[i] <= final_nodes[i-1]:
            final_nodes[i] = final_nodes[i-1] + 1e-12
            
    return {"nodes": final_nodes, "weights": final_weights}
