"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
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

def _true_integral(z_max, Om0, Ol0, w0, w1):
    """High-accuracy adaptive integration for reference."""
    result, _ = quad(lambda z: _integrand(z, Om0, Ol0, w0, w1), 0, z_max, limit=200)
    return result

def construct():
    t_start = time.time()
    deadline = t_start + 85.0  # Safety margin before 90s
    
    # Design cosmologies: span the expected parameter space
    design_cosmos = [
        (0.3, 0.7, -1.0, 0.0),    # Lambda CDM
        (0.27, 0.73, -1.0, 0.0),  # Planck-like
        (0.3, 0.7, -1.0, 1.0),    # CPL positive w1
        (0.3, 0.7, -1.0, -1.0),   # CPL negative w1
        (0.2, 0.8, -1.2, 0.5),    # More DE, phantom-ish w0
        (0.35, 0.65, -0.8, 0.3),  # Less DE
        (0.25, 0.75, -1.5, 1.0),  # Phantom w0, positive w1
        (0.3, 0.7, -0.5, -0.5),   # Mildly varying
        (0.3, 0.7, -1.0, 0.5),    # Moderate CPL
        (0.3, 0.7, -1.5, -0.5),   # Phantom, negative w1
    ]
    z_max_list = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    # Precompute true integrals for design set
    true_vals = []
    for (Om0, Ol0, w0, w1) in design_cosmos:
        for z_max in z_max_list:
            tv = _true_integral(z_max, Om0, Ol0, w0, w1)
            true_vals.append((Om0, Ol0, w0, w1, z_max, tv))
    
    n_true = len(true_vals)
    
    def _eval_quadrature(nodes, weights, Om0, Ol0, w0, w1, z_max):
        """Vectorized quadrature evaluation."""
        z_eval = z_max * nodes
        f_vals = np.array([_integrand(z_eval[j], Om0, Ol0, w0, w1) for j in range(8)])
        return z_max * np.dot(weights, f_vals)
    
    def _objective(params):
        """Worst relative error using cumulative node parameterization."""
        # Parameterize nodes via cumulative log-spacings to ensure strict monotonicity
        # nodes[0] = exp(log_gap_0) * 0.01, nodes[i] = nodes[i-1] + exp(log_gap_i) * (1 - nodes[i-1])
        log_gaps = params[:8]
        log_ws = params[8:]
        
        # Build nodes from cumulative gaps
        nodes = np.zeros(8)
        nodes[0] = 0.01 * np.exp(log_gaps[0] * 0.5)  # First node in (0, 0.5)
        for i in range(1, 8):
            gap = np.exp(log_gaps[i] * 0.5) * (1.0 - nodes[i-1]) * 0.5
            nodes[i] = nodes[i-1] + gap
        
        # Ensure nodes are in (0,1) and strictly increasing
        if nodes[0] <= 0 or nodes[-1] >= 1 or np.any(np.diff(nodes) <= 1e-15):
            return 1e6
        
        weights = np.exp(log_ws)
        weights = weights / np.sum(weights)
        
        max_err = 0.0
        for i in range(n_true):
            Om0, Ol0, w0, w1, z_max, tv = true_vals[i]
            if abs(tv) < 1e-15:
                continue
            approx = _eval_quadrature(nodes, weights, Om0, Ol0, w0, w1, z_max)
            rel_err = abs(approx - tv) / abs(tv) * 100.0
            if rel_err > max_err:
                max_err = rel_err
        return max_err
    
    # Initial guess: Gauss-Legendre mapped to [0,1]
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    gl_nodes_01 = 0.5 * (gl_nodes + 1.0)
    
    # Convert GL nodes to log-gap parameterization
    def _nodes_to_log_gaps(nodes):
        log_gaps = np.zeros(8)
        log_gaps[0] = 2.0 * np.log(nodes[0] / 0.01)
        for i in range(1, 8):
            gap = nodes[i] - nodes[i-1]
            frac = gap / (1.0 - nodes[i-1])
            if frac > 0:
                log_gaps[i] = 2.0 * np.log(frac / 0.5)
            else:
                log_gaps[i] = -10.0
        return log_gaps
    
    init_log_gaps = _nodes_to_log_gaps(gl_nodes_01)
    init_log_ws = np.log(gl_weights)
    x0 = np.concatenate([init_log_gaps, init_log_ws])
    
    best_params = x0.copy()
    best_score = _objective(x0)
    
    # Differential evolution for global search
    bounds = [(-5, 5)] * 8 + [(-5, 5)] * 8
    
    if time.time() < deadline - 30:
        try:
            de_result = differential_evolution(
                _objective, bounds,
                seed=42, maxiter=50, popsize=15,
                tol=1e-8, mutation=(0.3, 1.0), recombination=0.8,
                polish=False, updating='immediate'
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_params = de_result.x
        except Exception:
            pass
    
    # Local refinement with Nelder-Mead
    if time.time() < deadline - 10:
        try:
            nm_result = minimize(_objective, best_params, method='Nelder-Mead',
                                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            if nm_result.fun < best_score:
                best_score = nm_result.fun
                best_params = nm_result.x
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 5:
        try:
            nm_result2 = minimize(_objective, best_params, method='Nelder-Mead',
                                 options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            if nm_result2.fun < best_score:
                best_score = nm_result2.fun
                best_params = nm_result2.x
        except Exception:
            pass
    
    # Extract final nodes and weights
    log_gaps = best_params[:8]
    log_ws = best_params[8:]
    
    nodes = np.zeros(8)
    nodes[0] = 0.01 * np.exp(log_gaps[0] * 0.5)
    for i in range(1, 8):
        gap = np.exp(log_gaps[i] * 0.5) * (1.0 - nodes[i-1]) * 0.5
        nodes[i] = nodes[i-1] + gap
    
    weights = np.exp(log_ws)
    weights = weights / np.sum(weights)
    
    # Ensure nodes are strictly in (0,1) and increasing
    nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
    order = np.argsort(nodes)
    nodes = nodes[order]
    weights = weights[order]
    
    # Normalize weights to sum to 1
    weights = weights / np.sum(weights)
    
    # Ensure positive
    weights = np.maximum(weights, 1e-15)
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
