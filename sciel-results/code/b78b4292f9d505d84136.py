"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, linprog
from scipy.integrate import quad

def _E_of_z_vec(z, Om0, Ol0, w0, w1):
    """E(z) = H(z)/H0 for flat CPL model. Vectorized."""
    a = 1.0 / (1.0 + z)
    a3 = a ** 3
    exponent = -3.0 * w1 * (1.0 - a)
    de_factor = a ** (-3.0 * (1.0 + w0)) * np.exp(exponent)
    E2 = Om0 / a3 + Ol0 * de_factor
    return np.sqrt(np.maximum(E2, 1e-300))

def construct():
    t_start = time.time()
    deadline = t_start + 82.0

    # Design cosmologies: broader coverage of parameter space
    design_cosmos = [
        (0.30, 0.70, -1.00, 0.00),   # Lambda CDM
        (0.27, 0.73, -1.00, 0.00),   # Planck-like
        (0.30, 0.70, -1.00, 1.00),   # CPL positive w1
        (0.30, 0.70, -1.00, -1.00),  # CPL negative w1
        (0.20, 0.80, -1.20, 0.50),   # More DE
        (0.35, 0.65, -0.80, 0.30),   # Less DE
        (0.25, 0.75, -1.50, 1.00),   # Phantom w0
        (0.30, 0.70, -0.50, -0.50),  # Mildly varying
        (0.30, 0.70, -1.00, 0.50),   # Moderate CPL
        (0.30, 0.70, -1.50, -0.50),  # Phantom, negative w1
        (0.30, 0.70, -0.80, -1.00),  # Another variant
        (0.25, 0.75, -1.00, 0.80),   # Moderate DE, positive w1
        (0.30, 0.70, -1.00, -0.50),  # CPL negative w1 moderate
        (0.28, 0.72, -0.90, 0.20),   # Near Planck, mild CPL
        (0.32, 0.68, -1.10, -0.30),  # Slightly more matter
        (0.30, 0.70, -1.00, 2.00),   # Strong CPL positive
        (0.30, 0.70, -1.00, -2.00),  # Strong CPL negative
        (0.25, 0.75, -0.70, 0.00),   # Less negative w0
        (0.35, 0.65, -1.30, 0.00),   # More negative w0
        (0.30, 0.70, -1.00, 0.30),   # Mild positive w1
    ]
    z_max_list = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    
    # Precompute true integrals
    true_vals = []
    for (Om0, Ol0, w0, w1) in design_cosmos:
        for z_max in z_max_list:
            def integrand_u(u, z_max=z_max, Om0=Om0, Ol0=Ol0, w0=w0, w1=w1):
                z_val = z_max * u
                if z_val < 1e-15:
                    return 1.0
                return 1.0 / _E_of_z_vec(z_val, Om0, Ol0, w0, w1)
            
            result, _ = quad(integrand_u, 0, 1, limit=200)
            true_val = z_max * result
            true_vals.append((Om0, Ol0, w0, w1, z_max, true_val))
    
    # Vectorize: precompute arrays
    n_design = len(true_vals)
    true_vals_arr = np.array([tv[5] for tv in true_vals])
    cosmos_params = np.array([tv[:4] for tv in true_vals])
    z_max_arr = np.array([tv[4] for tv in true_vals])
    
    def _soft_max(errors, beta=30.0):
        """Smooth max via log-sum-exp."""
        e = errors * beta
        e_max = np.max(e)
        if e_max > 700:
            e_max = 700.0
        s = e - e_max
        return e_max / beta + np.log(np.sum(np.exp(s))) / beta
    
    def _compute_rel_error(nodes, weights):
        """Compute relative errors for all design points."""
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
        return rel_err
    
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
        
        rel_err = _compute_rel_error(nodes, weights)
        return _soft_max(rel_err, beta=30.0)
    
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
        
        rel_err = _compute_rel_error(nodes, weights)
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
    
    # Structural approach: basis expansion + convex optimization
    # Model 1/E(z) as a linear combination of basis functions
    # The integral becomes sum_i w_i * sum_k c_k * phi_k(z_i)
    # where phi_k are basis functions spanning the DE variation
    
    # Phase 1: Find good nodes via basis fitting
    # Use Chebyshev polynomials as basis on [0,1]
    def _cheb_basis(u, n_terms=6):
        """Chebyshev polynomials T_0..T_{n_terms-1} evaluated at u in [0,1]."""
        x = 2.0 * u - 1.0  # Map to [-1,1]
        basis = np.zeros((len(u), n_terms))
        for k in range(n_terms):
            if k == 0:
                basis[:, k] = 1.0
            elif k == 1:
                basis[:, k] = x
            else:
                basis[:, k] = 2.0 * x * basis[:, k-1] - basis[:, k-2]
        return basis
    
    # For each cosmology, the integrand 1/E(z) can be approximated
    # by a weighted sum of basis functions. The weights depend on cosmology.
    # The key insight: find nodes such that the quadrature is exact
    # for a set of basis functions that span the variation.
    
    # Strategy: 
    # 1. Sample 1/E(z) at many z for each cosmology
    # 2. Fit to Chebyshev basis to get coefficients
    # 3. Find nodes that minimize worst-case error over all cosmologies
    
    # Precompute basis coefficients for each cosmology
    n_terms = 6
    z_sample = np.linspace(0.001, 5.0, 500)
    
    coeff_matrix = np.zeros((n_design, n_terms))
    for j in range(n_design):
        Om0, Ol0, w0, w1 = cosmos_params[j]
        z_eval = z_sample
        E = _E_of_z_vec(z_eval, Om0, Ol0, w0, w1)
        inv_E = 1.0 / E
        
        # Fit to Chebyshev basis
        u_sample = z_eval / 5.0  # Normalize to [0,1]
        basis = _cheb_basis(u_sample, n_terms)
        # Least squares fit
        coeffs, _, _, _ = np.linalg.lstsq(basis, inv_E, rcond=None)
        coeff_matrix[j, :] = coeffs
    
    # Now optimize nodes to minimize worst-case error
    # The approximation at node z_i is sum_k coeff_jk * T_k(z_i/5)
    # But we need the actual integral, not just point values
    
    # Better approach: direct optimization with the structural insight
    # Initialize with Gauss-Legendre nodes
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
                         options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
        h = _objective_hard(result.x)
        if h < best_score:
            best_score = h
            best_params = result.x.copy()
    except Exception:
        pass
    
    if time.time() > deadline:
        nodes, weights = _extract(best_params)
        return {"nodes": nodes, "weights": weights}
    
    # Phase 2: L-BFGS-B refinement
    try:
        result2 = minimize(_objective, best_params, method='L-BFGS-B',
                          options={'maxiter': 1500, 'gtol': 1e-14, 'ftol': 1e-14})
        h = _objective_hard(result2.x)
        if h < best_score:
            best_score = h
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
    for p in [0.5, 2.0, 3.0]:
        pow_nodes = (np.arange(1, 9) / 8.0) ** p
        pow_nodes = np.clip(pow_nodes, 1e-6, 1 - 1e-6)
        if np.all(np.diff(pow_nodes) > 0):
            pow_raw = np.arctanh(2.0 * pow_nodes - 1.0)
            diverse_starts.append(np.concatenate([pow_raw, init_log_w]))
    
    # Gauss-Lobatto
    lob_nodes, lob_weights = np.polynomial.legendre.leggauss(8)
    lob_nodes_01 = 0.5 * (lob_nodes + 1.0)
    lob_raw = np.arctanh(2.0 * lob_nodes_01 - 1.0)
    diverse_starts.append(np.concatenate([lob_raw, init_log_w]))
    
    # Random perturbations
    for i in range(8):
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
                              options={'maxiter': 800, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
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
                              options={'maxiter': 5000, 'xatol': 1e-16, 'fatol': 1e-16, 'adaptive': True})
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
                              options={'maxiter': 3000, 'gtol': 1e-15, 'ftol': 1e-15})
            h = _objective_hard(result_g.x)
            if h < best_score:
                best_score = h
                best_params = result_g.x.copy()
        except Exception:
            pass
    
    # Final fine-tuning: try small perturbations around best
    if time.time() < deadline - 0.5:
        for _ in range(5):
            x0_t = best_params.copy()
            x0_t[:8] += np.random.randn(8) * 0.01
            x0_t[8:] += np.random.randn(8) * 0.005
            try:
                result_t = minimize(_objective, x0_t, method='Nelder-Mead',
                                  options={'maxiter': 500, 'xatol': 1e-15, 'fatol': 1e-15, 'adaptive': True})
                h = _objective_hard(result_t.x)
                if h < best_score:
                    best_score = h
                    best_params = result_t.x.copy()
            except Exception:
                pass
    
    nodes, weights = _extract(best_params)
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
