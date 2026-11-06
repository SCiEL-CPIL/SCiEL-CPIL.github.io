"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad

def _E_of_z(z, Om0, Ol0, w0, w1):
    """E(z) = H(z)/H0 for flat CPL model."""
    a = 1.0 / (1.0 + z)
    a3 = a ** 3
    exponent = -3.0 * w1 * (1.0 - a)
    de_factor = a ** (-3.0 * (1.0 + w0)) * np.exp(exponent)
    E2 = Om0 / a3 + Ol0 * de_factor
    return np.sqrt(np.maximum(E2, 1e-300))

def _integrand(z, Om0, Ol0, w0, w1):
    """f(z) = z/(1+z) * 1/E(z)"""
    if z < 1e-15:
        return 0.0
    E = _E_of_z(z, Om0, Ol0, w0, w1)
    return z / (1.0 + z) / E

def construct():
    t_start = time.time()
    deadline = t_start + 82.0  # Safety margin before 90s kill

    # Design cosmologies and z values for optimization
    design_cosmos = [
        (0.3, 0.7, -1.0, 0.0),    # Lambda CDM
        (0.27, 0.73, -1.0, 0.0),  # Planck-like
        (0.3, 0.7, -1.0, 1.0),    # CPL positive w1
        (0.3, 0.7, -1.0, -1.0),   # CPL negative w1
        (0.2, 0.8, -1.2, 0.5),    # More DE
        (0.35, 0.65, -0.8, 0.3),  # Less DE
        (0.25, 0.75, -1.5, 1.0),  # Phantom w0
        (0.3, 0.7, -0.5, -0.5),   # Mildly varying
        (0.3, 0.7, -1.0, 0.5),    # Moderate CPL
        (0.3, 0.7, -1.5, -0.5),   # Phantom, negative w1
        (0.3, 0.7, -0.8, -1.0),   # Another variant
        (0.25, 0.75, -1.0, 0.8),  # Moderate DE, positive w1
    ]
    z_max_list = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    # Precompute true integrals
    true_vals = []
    for (Om0, Ol0, w0, w1) in design_cosmos:
        for z_max in z_max_list:
            def integrand_u(u, z_max=z_max, Om0=Om0, Ol0=Ol0, w0=w0, w1=w1):
                z_val = z_max * u
                if z_val < 1e-15:
                    return 1.0
                return 1.0 / _E_of_z(z_val, Om0, Ol0, w0, w1)
            
            result, _ = quad(integrand_u, 0, 1, limit=200)
            true_val = z_max * result
            true_vals.append((Om0, Ol0, w0, w1, z_max, true_val))
    
    # Vectorize: precompute arrays
    n_design = len(true_vals)
    true_vals_arr = np.array([tv[5] for tv in true_vals])
    cosmos_params = np.array([tv[:4] for tv in true_vals])
    z_max_arr = np.array([tv[4] for tv in true_vals])
    
    def _soft_max(errors, beta=50.0):
        """Smooth max via log-sum-exp."""
        e = errors * beta
        e_max = np.max(e)
        if e_max > 700:
            e_max = 700.0
        s = e - e_max
        return e_max / beta + np.log(np.sum(np.exp(s))) / beta
    
    def _objective(params):
        n = 8
        raw = params[:n]
        log_w = params[n:]
        
        nodes = 0.5 + 0.5 * np.tanh(raw)
        if np.any(np.diff(nodes) <= 1e-14):
            return 1e10
        
        weights = np.exp(log_w)
        w_sum = np.sum(weights)
        if w_sum < 1e-300:
            return 1e10
        weights = weights / w_sum
        
        # Vectorized computation of all approximations
        # For each design point j and node i: E(z_max_j * nodes_i)
        # Shape: (n_design, 8)
        z_eval = z_max_arr[:, None] * nodes[None, :]  # (n_design, 8)
        
        # Compute E for all evaluations
        a = 1.0 / (1.0 + z_eval)
        a3 = a ** 3
        exponent = -3.0 * cosmos_params[:, 3:4] * (1.0 - a)
        de_factor = a ** (-3.0 * (1.0 + cosmos_params[:, 2:3])) * np.exp(exponent)
        E2 = cosmos_params[:, 0:1] / a3 + cosmos_params[:, 1:2] * de_factor
        E = np.sqrt(np.maximum(E2, 1e-300))
        
        # approx_j = z_max_j * sum_i weights_i / E_ji
        inv_E = 1.0 / E  # (n_design, 8)
        approx = z_max_arr * (inv_E @ weights)  # (n_design,)
        
        rel_err = np.abs(approx - true_vals_arr) / np.maximum(np.abs(true_vals_arr), 1e-300) * 100.0
        
        # Use soft-max for smoothness
        return _soft_max(rel_err, beta=50.0)
    
    def _objective_hard(params):
        """Exact max relative error for final evaluation."""
        n = 8
        raw = params[:n]
        log_w = params[n:]
        
        nodes = 0.5 + 0.5 * np.tanh(raw)
        if np.any(np.diff(nodes) <= 1e-14):
            return 1e10
        
        weights = np.exp(log_w)
        w_sum = np.sum(weights)
        if w_sum < 1e-300:
            return 1e10
        weights = weights / w_sum
        
        z_eval = z_max_arr[:, None] * nodes[None, :]
        a = 1.0 / (1.0 + z_eval)
        a3 = a ** 3
        exponent = -3.0 * cosmos_params[:, 3:4] * (1.0 - a)
        de_factor = a ** (-3.0 * (1.0 + cosmos_params[:, 2:3])) * np.exp(exponent)
        E2 = cosmos_params[:, 0:1] / a3 + cosmos_params[:, 1:2] * de_factor
        E = np.sqrt(np.maximum(E2, 1e-300))
        
        inv_E = 1.0 / E
        approx = z_max_arr * (inv_E @ weights)
        
        rel_err = np.abs(approx - true_vals_arr) / np.maximum(np.abs(true_vals_arr), 1e-300) * 100.0
        return np.max(rel_err)
    
    def _extract(params):
        raw = params[:8]
        log_w = params[8:]
        nodes = 0.5 + 0.5 * np.tanh(raw)
        weights = np.exp(log_w)
        weights = weights / np.sum(weights)
        nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
        order = np.argsort(nodes)
        nodes = nodes[order]
        weights = weights[order]
        weights = np.maximum(weights, 1e-15)
        weights = weights / np.sum(weights)
        return nodes, weights
    
    # Initial guess: Gauss-Legendre
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    gl_nodes_01 = 0.5 * (gl_nodes + 1.0)
    
    init_raw = np.arctanh(2.0 * gl_nodes_01 - 1.0)
    init_log_w = np.log(gl_weights)
    x0 = np.concatenate([init_raw, init_log_w])
    
    best_params = x0.copy()
    best_score = _objective_hard(x0)
    
    # Phase 1: Nelder-Mead with soft objective
    try:
        result = minimize(_objective, x0, method='Nelder-Mead',
                         options={'maxiter': 1500, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
        if _objective_hard(result.x) < best_score:
            best_score = _objective_hard(result.x)
            best_params = result.x.copy()
    except Exception:
        pass
    
    if time.time() > deadline:
        nodes, weights = _extract(best_params)
        return {"nodes": nodes, "weights": weights}
    
    # Phase 2: L-BFGS-B refinement
    try:
        result2 = minimize(_objective, best_params, method='L-BFGS-B',
                          options={'maxiter': 1000, 'gtol': 1e-14, 'ftol': 1e-14})
        if _objective_hard(result2.x) < best_score:
            best_score = _objective_hard(result2.x)
            best_params = result2.x.copy()
    except Exception:
        pass
    
    if time.time() > deadline:
        nodes, weights = _extract(best_params)
        return {"nodes": nodes, "weights": weights}
    
    # Phase 3: Multi-start with diverse initializations
    np.random.seed(42)
    diverse_starts = [x0.copy()]
    
    # Chebyshev nodes
    cheb_nodes = 0.5 - 0.5 * np.cos(np.pi * (np.arange(8) + 0.5) / 8)
    cheb_raw = np.arctanh(2.0 * cheb_nodes - 1.0)
    diverse_starts.append(np.concatenate([cheb_raw, init_log_w]))
    
    # Log-spaced
    log_nodes = np.logspace(-2, 0, 8)
    log_nodes = log_nodes / log_nodes[-1]
    log_nodes = np.clip(log_nodes, 1e-6, 1 - 1e-6)
    if np.all(np.diff(log_nodes) > 0):
        log_raw = np.arctanh(2.0 * log_nodes - 1.0)
        diverse_starts.append(np.concatenate([log_raw, init_log_w]))
    
    # Power-law
    for p in [0.5, 2.0]:
        pow_nodes = (np.arange(1, 9) / 8.0) ** p
        pow_nodes = np.clip(pow_nodes, 1e-6, 1 - 1e-6)
        if np.all(np.diff(pow_nodes) > 0):
            pow_raw = np.arctanh(2.0 * pow_nodes - 1.0)
            diverse_starts.append(np.concatenate([pow_raw, init_log_w]))
    
    # Random perturbations
    for i in range(6):
        if time.time() > deadline:
            break
        x0_r = x0.copy()
        x0_r[:8] += np.random.randn(8) * 0.3
        x0_r[8:] += np.random.randn(8) * 0.15
        diverse_starts.append(x0_r)
    
    for x0_r in diverse_starts:
        if time.time() > deadline - 5:
            break
        try:
            result_r = minimize(_objective, x0_r, method='Nelder-Mead',
                              options={'maxiter': 600, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
            h = _objective_hard(result_r.x)
            if h < best_score:
                best_score = h
                best_params = result_r.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 3:
        try:
            result_f = minimize(_objective, best_params, method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-16, 'fatol': 1e-16, 'adaptive': True})
            h = _objective_hard(result_f.x)
            if h < best_score:
                best_score = h
                best_params = result_f.x.copy()
        except Exception:
            pass
    
    # One more L-BFGS-B polish
    if time.time() < deadline - 1:
        try:
            result_g = minimize(_objective, best_params, method='L-BFGS-B',
                              options={'maxiter': 2000, 'gtol': 1e-15, 'ftol': 1e-15})
            h = _objective_hard(result_g.x)
            if h < best_score:
                best_score = h
                best_params = result_g.x.copy()
        except Exception:
            pass
    
    nodes, weights = _extract(best_params)
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
