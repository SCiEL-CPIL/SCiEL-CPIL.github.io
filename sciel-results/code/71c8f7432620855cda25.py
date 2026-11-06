"""
Cosmology: universal luminosity-distance quadrature (8-node positive rule).
Combines the alternating optimization strategy (LP for weights, Nelder-Mead for nodes)
with a robust design space and vectorized evaluation.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog
from scipy.integrate import quad

_START = time.time()
_DEADLINE = _START + 85.0

def _time_left():
    return _DEADLINE - time.time()

def _E_of_z_vec(z, Om0, Ol0, w0, w1):
    """E(z) = H(z)/H0 for flat CPL model. Vectorized."""
    a = 1.0 / (1.0 + z)
    a3 = a ** 3
    exponent = -3.0 * w1 * (1.0 - a)
    de_factor = a ** (-3.0 * (1.0 + w0)) * np.exp(exponent)
    E2 = Om0 / a3 + Ol0 * de_factor
    return np.sqrt(np.maximum(E2, 1e-300))

def construct():
    # Design cosmologies covering the expected parameter space
    # Om0, Ol0, w0, w1
    design_cosmos = [
        (0.30, 0.70, -1.00,  0.00),  # Lambda CDM
        (0.27, 0.73, -1.00,  0.00),  # Planck-like
        (0.30, 0.70, -1.00,  1.00),  # CPL positive w1
        (0.30, 0.70, -1.00, -1.00),  # CPL negative w1
        (0.20, 0.80, -1.20,  0.50),  # More DE
        (0.35, 0.65, -0.80,  0.30),  # Less DE
        (0.25, 0.75, -1.50,  1.00),  # Phantom w0
        (0.30, 0.70, -0.50, -0.50),  # Mildly varying
        (0.30, 0.70, -1.00,  0.50),  # Moderate CPL
        (0.30, 0.70, -1.50, -0.50),  # Phantom, negative w1
        (0.30, 0.70, -0.80, -1.00),  # Another variant
        (0.25, 0.75, -1.00,  0.80),  # Moderate DE, positive w1
        (0.28, 0.72, -0.90,  0.20),  # Near Planck, mild CPL
        (0.32, 0.68, -1.10, -0.30),  # Slightly more matter
        (0.30, 0.70, -1.00,  2.00),  # Strong CPL positive
        (0.30, 0.70, -1.00, -2.00),  # Strong CPL negative
        (0.25, 0.75, -0.70,  0.00),  # Less negative w0
        (0.35, 0.65, -1.30,  0.00),  # More negative w0
        (0.30, 0.70, -1.00,  0.30),  # Mild positive w1
        (0.30, 0.70, -1.00, -0.30),  # Mild negative w1
    ]
    
    # z_max values to cover the integration range
    # The quadrature is for integral from 0 to 1 in u-space, scaled by z_max.
    # We test for a range of z_max to ensure universality.
    z_max_list = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    # Precompute true integrals for design points
    true_vals = []
    cosmos_params = []
    z_max_arr_list = []
    
    for (Om0, Ol0, w0, w1) in design_cosmos:
        for z_max in z_max_list:
            def integrand_u(u, z_max=z_max, Om0=Om0, Ol0=Ol0, w0=w0, w1=w1):
                z_val = z_max * u
                if z_val < 1e-15:
                    return 1.0
                return 1.0 / _E_of_z_vec(z_val, Om0, Ol0, w0, w1)
            
            result, _ = quad(integrand_u, 0, 1, limit=200, epsabs=1e-12, epsrel=1e-12)
            true_vals.append(result)
            cosmos_params.append((Om0, Ol0, w0, w1))
            z_max_arr_list.append(z_max)
            
    true_vals_arr = np.array(true_vals)
    cosmos_params = np.array(cosmos_params)
    z_max_arr = np.array(z_max_arr_list)
    n_design = len(true_vals_arr)

    def _compute_inv_E(nodes):
        """Compute 1/E for all design points and given nodes. Returns (n_design, 8)."""
        # nodes: (8,)
        # z_eval: (n_design, 8)
        z_eval = z_max_arr[:, None] * nodes[None, :]
        a = 1.0 / (1.0 + z_eval)
        a3 = a ** 3
        # Vectorized E(z) calculation
        # cosmos_params: (n_design, 4)
        # Need to broadcast w0, w1, Om0, Ol0
        Om0 = cosmos_params[:, 0:1]
        Ol0 = cosmos_params[:, 1:2]
        w0 = cosmos_params[:, 2:3]
        w1 = cosmos_params[:, 3:4]
        
        exponent = -3.0 * w1 * (1.0 - a)
        de_factor = a ** (-3.0 * (1.0 + w0)) * np.exp(exponent)
        E2 = Om0 / a3 + Ol0 * de_factor
        E = np.sqrt(np.maximum(E2, 1e-300))
        return 1.0 / E

    def _solve_weights_lp(nodes):
        """
        Solve for optimal weights minimizing worst-case relative error via LP.
        Minimize t s.t. |V w - J|_inf / J <= t, sum(w) = 1, w >= 0.
        V: (n_design, 8) matrix of 1/E(z_max_j * n_i)
        J: (n_design,) vector of true integrals
        """
        V = _compute_inv_E(nodes)
        J = true_vals_arr
        
        # Variables: w[0..7], t
        # Minimize t
        c = np.zeros(9)
        c[8] = 1.0
        
        # Constraints:
        # V w - J t <= J  =>  V w - J t - J <= 0
        # -V w - J t <= -J => -V w - J t + J <= 0
        
        A_ub = np.zeros((2 * n_design, 9))
        b_ub = np.zeros(2 * n_design)
        
        # First set: V w - J t <= J
        A_ub[:n_design, :8] = V
        A_ub[:n_design, 8] = -J
        b_ub[:n_design] = J
        
        # Second set: -V w - J t <= -J
        A_ub[n_design:, :8] = -V
        A_ub[n_design:, 8] = -J
        b_ub[n_design:] = -J
        
        # Equality: sum(w) = 1
        A_eq = np.zeros((1, 9))
        A_eq[0, :8] = 1.0
        b_eq = np.array([1.0])
        
        # Bounds: w >= 0, t >= 0
        bounds = [(0, None)] * 8 + [(0, None)]
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method='highs')
            if res.success:
                weights = res.x[:8]
                weights = np.maximum(weights, 0)
                weights = weights / np.sum(weights)
                return weights
            else:
                # Fallback: Gauss-Legendre weights
                _, gl_weights = np.polynomial.legendre.leggauss(8)
                return gl_weights
        except Exception:
            _, gl_weights = np.polynomial.legendre.leggauss(8)
            return gl_weights

    def _worst_rel_error(nodes, weights):
        """Compute worst relative error in percent."""
        V = _compute_inv_E(nodes)
        approx = V @ weights
        rel_err = np.abs(approx - true_vals_arr) / np.maximum(np.abs(true_vals_arr), 1e-300) * 100.0
        return np.max(rel_err)

    def _nodes_objective(raw_nodes):
        """Objective for node optimization."""
        nodes = 0.5 + 0.5 * np.tanh(raw_nodes)
        nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
        order = np.argsort(nodes)
        nodes = nodes[order]
        
        if np.any(np.diff(nodes) <= 1e-12):
            return 1e10
        
        weights = _solve_weights_lp(nodes)
        return _worst_rel_error(nodes, weights)

    # Initial node guesses
    init_nodes_list = []
    
    # Chebyshev nodes
    k = np.arange(8)
    cheb_nodes = 0.5 * (1.0 - np.cos(np.pi * (k + 0.5) / 8))
    init_nodes_list.append(cheb_nodes)
    
    # Gauss-Legendre nodes
    gl_nodes, _ = np.polynomial.legendre.leggauss(8)
    gl_01 = 0.5 * (gl_nodes + 1.0)
    init_nodes_list.append(gl_01)
    
    # Uniform nodes
    unif_nodes = np.linspace(0.05, 0.95, 8)
    init_nodes_list.append(unif_nodes)
    
    # Log-spaced nodes
    log_nodes = np.logspace(-2, 0, 8)
    log_nodes = log_nodes / log_nodes[-1]
    log_nodes = np.clip(log_nodes, 1e-6, 1 - 1e-6)
    init_nodes_list.append(log_nodes)
    
    # Random nodes
    rng = np.random.default_rng(42)
    rand_nodes = np.sort(rng.uniform(0.02, 0.98, 8))
    init_nodes_list.append(rand_nodes)

    best_nodes = None
    best_weights = None
    best_err = np.inf

    # Phase 1: Evaluate initial node sets
    for init_nodes in init_nodes_list:
        if _time_left() < 15:
            break
        weights = _solve_weights_lp(init_nodes)
        err = _worst_rel_error(init_nodes, weights)
        if err < best_err:
            best_err = err
            best_nodes = init_nodes.copy()
            best_weights = weights.copy()

    # Phase 2: Nelder-Mead refinement of nodes
    if _time_left() > 20 and best_nodes is not None:
        best_raw = np.arctanh(2.0 * best_nodes - 1.0)
        
        try:
            res_nm = minimize(_nodes_objective, best_raw, method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12,
                                      'adaptive': True})
            
            nodes = 0.5 + 0.5 * np.tanh(res_nm.x)
            nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
            order = np.argsort(nodes)
            nodes = nodes[order]
            weights = _solve_weights_lp(nodes)
            err = _worst_rel_error(nodes, weights)
            
            if err < best_err:
                best_err = err
                best_nodes = nodes.copy()
                best_weights = weights.copy()
        except Exception:
            pass

    # Phase 3: Additional random restarts if time permits
    if _time_left() > 30:
        for seed in [123, 456, 789, 321, 654, 987]:
            if _time_left() < 20:
                break
            rng2 = np.random.default_rng(seed)
            rand_n = np.sort(rng2.uniform(0.01, 0.99, 8))
            raw_n = np.arctanh(2.0 * rand_n - 1.0)
            
            try:
                res = minimize(_nodes_objective, raw_n, method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12,
                                      'adaptive': True})
                nodes = 0.5 + 0.5 * np.tanh(res.x)
                nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
                order = np.argsort(nodes)
                nodes = nodes[order]
                weights = _solve_weights_lp(nodes)
                err = _worst_rel_error(nodes, weights)
                
                if err < best_err:
                    best_err = err
                    best_nodes = nodes.copy()
                    best_weights = weights.copy()
            except Exception:
                pass

    # Final check and cleanup
    if best_nodes is None:
        # Fallback to Gauss-Legendre
        best_nodes, best_weights = np.polynomial.legendre.leggauss(8)
        best_nodes = 0.5 * (best_nodes + 1.0)
        best_weights = best_weights / np.sum(best_weights)

    # Ensure strict increase and bounds
    best_nodes = np.clip(best_nodes, 1e-10, 1.0 - 1e-10)
    order = np.argsort(best_nodes)
    best_nodes = best_nodes[order]
    best_weights = best_weights[order]
    best_weights = np.maximum(best_weights, 1e-15)
    best_weights = best_weights / np.sum(best_weights)

    return {"nodes": best_nodes, "weights": best_weights}

from scipy.optimize import minimize
# EVOLVE-BLOCK-END
