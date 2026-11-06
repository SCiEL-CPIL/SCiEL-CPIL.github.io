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

def construct():
    t_start = time.time()
    deadline = t_start + 85.0  # Safety margin before 90s kill

    # Design cosmologies: systematic grid covering parameter space
    # Om0 in [0.2, 0.4], w0 in [-1.5, -0.5], w1 in [-1.0, 1.0]
    Om0_vals = [0.2, 0.25, 0.3, 0.35, 0.4]
    w0_vals = [-1.5, -1.2, -1.0, -0.8, -0.5]
    w1_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    design_cosmos = []
    for Om0 in Om0_vals:
        Ol0 = 1.0 - Om0
        for w0 in w0_vals:
            for w1 in w1_vals:
                design_cosmos.append((Om0, Ol0, w0, w1))
    
    # Reduce to a manageable subset - use stratified sampling
    np.random.seed(12345)
    if len(design_cosmos) > 40:
        indices = np.random.choice(len(design_cosmos), 40, replace=False)
        design_cosmos = [design_cosmos[i] for i in indices]
    
    # Ensure we have key reference cosmologies
    key_cosmos = [
        (0.3, 0.7, -1.0, 0.0),    # Lambda CDM
        (0.27, 0.73, -1.0, 0.0),  # Planck-like
        (0.3, 0.7, -1.0, 1.0),    # CPL positive w1
        (0.3, 0.7, -1.0, -1.0),   # CPL negative w1
        (0.2, 0.8, -1.2, 0.5),    # More DE
        (0.35, 0.65, -0.8, 0.3),  # Less DE
        (0.25, 0.75, -1.5, 1.0),  # Phantom w0
        (0.3, 0.7, -0.5, -0.5),   # Mildly varying
    ]
    for kc in key_cosmos:
        if kc not in design_cosmos:
            design_cosmos.append(kc)
    
    z_max_list = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    # Precompute true integrals using high-accuracy quad
    true_vals = []
    for (Om0, Ol0, w0, w1) in design_cosmos:
        for z_max in z_max_list:
            def integrand_u(u, z_max=z_max, Om0=Om0, Ol0=Ol0, w0=w0, w1=w1):
                z_val = z_max * u
                if z_val < 1e-15:
                    return 1.0
                return 1.0 / _E_of_z(z_val, Om0, Ol0, w0, w1)
            
            result, _ = quad(integrand_u, 0, 1, limit=500, epsabs=1e-14, epsrel=1e-14)
            true_val = z_max * result
            true_vals.append((Om0, Ol0, w0, w1, z_max, true_val))
    
    n_design = len(true_vals)
    
    # Objective: worst relative error, with smooth surrogate option
    def _objective(params, smooth_beta=0.0):
        n = 8
        raw_nodes = params[:n]
        log_w = params[n:]
        
        # Nodes: use sigmoid of cumulative sum for guaranteed monotonicity
        # Or use tanh and check
        nodes = 0.5 + 0.5 * np.tanh(raw_nodes)
        
        # Check strictly increasing
        diffs = np.diff(nodes)
        if np.any(diffs <= 1e-12):
            return 1e10
        
        # Weights: exp and normalize
        weights = np.exp(log_w)
        w_sum = np.sum(weights)
        if w_sum < 1e-300:
            return 1e10
        weights = weights / w_sum
        
        # Compute relative errors
        rel_errors = np.zeros(n_design)
        for idx, (Om0, Ol0, w0, w1, z_max, tv) in enumerate(true_vals):
            if abs(tv) < 1e-15:
                rel_errors[idx] = 0.0
                continue
            # Approximation: z_max * sum_i w_i / E(z_max * t_i)
            z_nodes = z_max * nodes
            E_vals = np.array([_E_of_z(z_nodes[i], Om0, Ol0, w0, w1) for i in range(8)])
            approx = z_max * np.sum(weights / E_vals)
            rel_errors[idx] = abs(approx - tv) / abs(tv) * 100.0
        
        if smooth_beta > 0:
            # Soft-max using log-sum-exp for smoothness
            max_err = np.max(rel_errors)
            if max_err < 1e-10:
                return max_err
            # Log-sum-exp based soft max
            scaled = rel_errors / smooth_beta
            max_scaled = np.max(scaled)
            lse = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
            return lse * smooth_beta
        else:
            return np.max(rel_errors)
    
    # Initial guess: Gauss-Legendre
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    gl_nodes_01 = 0.5 * (gl_nodes + 1.0)
    
    init_raw = np.arctanh(2.0 * gl_nodes_01 - 1.0)
    init_log_w = np.log(gl_weights)
    x0 = np.concatenate([init_raw, init_log_w])
    
    best_params = x0.copy()
    best_score = _objective(x0)
    
    # Phase 1: Nelder-Mead with smooth objective for better convergence
    if time.time() < deadline - 30:
        try:
            result = minimize(lambda p: _objective(p, smooth_beta=5.0), x0, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            if result.fun < best_score + 0.1:  # Allow some tolerance
                # Verify with exact objective
                exact_score = _objective(result.x)
                if exact_score < best_score:
                    best_score = exact_score
                    best_params = result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead with exact objective
    if time.time() < deadline - 20:
        try:
            result2 = minimize(_objective, best_params, method='Nelder-Mead',
                              options={'maxiter': 800, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            if result2.fun < best_score:
                best_score = result2.fun
                best_params = result2.x.copy()
        except Exception:
            pass
    
    # Phase 3: Multi-start with different initializations
    np.random.seed(42)
    for restart in range(10):
        if time.time() > deadline - 10:
            break
        
        # Different initializations
        if restart == 0:
            # Slightly perturbed GL
            x0_r = x0.copy()
            x0_r[:8] += np.random.randn(8) * 0.3
            x0_r[8:] += np.random.randn(8) * 0.1
        elif restart == 1:
            # Equal spacing nodes
            eq_nodes = np.linspace(0.1, 0.9, 8)
            eq_raw = np.arctanh(2.0 * eq_nodes - 1.0)
            eq_log_w = np.log(np.ones(8) / 8.0)
            x0_r = np.concatenate([eq_raw, eq_log_w])
        elif restart == 2:
            # Chebyshev-like nodes
            k = np.arange(8)
            chev_nodes = 0.5 * (1 + np.cos(np.pi * (k + 0.5) / 8))
            chev_nodes = chev_nodes[::-1]  # ascending
            chev_raw = np.arctanh(2.0 * chev_nodes - 1.0)
            chev_log_w = np.log(np.ones(8) / 8.0)
            x0_r = np.concatenate([chev_raw, chev_log_w])
        elif restart == 3:
            # Log-spaced nodes
            log_nodes = np.logspace(np.log10(0.05), np.log10(0.95), 8)
            log_raw = np.arctanh(2.0 * log_nodes - 1.0)
            log_log_w = np.log(np.ones(8) / 8.0)
            x0_r = np.concatenate([log_raw, log_log_w])
        else:
            # Random perturbations of best
            x0_r = best_params.copy()
            x0_r[:8] += np.random.randn(8) * (0.5 - restart * 0.05)
            x0_r[8:] += np.random.randn(8) * (0.2 - restart * 0.02)
        
        try:
            result_r = minimize(_objective, x0_r, method='Nelder-Mead',
                              options={'maxiter': 400, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            if result_r.fun < best_score:
                best_score = result_r.fun
                best_params = result_r.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 5:
        try:
            result_f = minimize(_objective, best_params, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
            if result_f.fun < best_score:
                best_score = result_f.fun
                best_params = result_f.x.copy()
        except Exception:
            pass
    
    # Extract final result
    raw = best_params[:8]
    log_w = best_params[8:]
    nodes = 0.5 + 0.5 * np.tanh(raw)
    weights = np.exp(log_w)
    weights = weights / np.sum(weights)
    
    # Ensure valid: strictly increasing nodes in (0,1), positive weights summing to 1
    nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
    order = np.argsort(nodes)
    nodes = nodes[order]
    weights = weights[order]
    weights = np.maximum(weights, 1e-15)
    weights = weights / np.sum(weights)
    
    # Final verification
    final_score = _objective(np.concatenate([np.arctanh(2.0 * nodes - 1.0), np.log(weights)]))
    if final_score > 1e6:
        # Fallback to Gauss-Legendre
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
        nodes = 0.5 * (gl_nodes + 1.0)
        weights = gl_weights
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
