"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import nnls
from scipy.integrate import quad
import time


def _E_cpl(a, w0, w1):
    """CPL dark energy: w(a) = w0 + w1*(1-a)
    E^2(a) = a^(-3) * (1 + (1+w0+w1)*(a^(-3*w1) - 1))
    """
    aw1 = a ** (-3.0 * w1)
    return np.sqrt(a ** (-3.0) * (1.0 + (1.0 + w0 + w1) * (aw1 - 1.0)))


def _E_lcdm(a, om_m):
    """LCDM: E^2 = om_m * a^(-3) + (1-om_m)"""
    return np.sqrt(om_m * a ** (-3.0) + (1.0 - om_m))


def _E_const_w(a, w):
    """Constant w: E^2 = a^(-3(1+w))"""
    return np.sqrt(a ** (-3.0 * (1.0 + w)))


def _E_exotic(a, p, q):
    """Exotic: E^2 = a^(-3) + p*a^(-q)"""
    return np.sqrt(a ** (-3.0) + p * a ** (-q))


def _exact_integral(z, E_func, *E_params):
    """Compute exact integral ∫_0^1 dx/E(1/(1+zx)) using adaptive quad."""
    def integrand(x):
        a_x = 1.0 / (1.0 + z * x)
        return 1.0 / E_func(a_x, *E_params)
    result, _ = quad(integrand, 0, 1, limit=300, epsabs=1e-14, epsrel=1e-14)
    return result


def _quad_value(z, nodes, weights, E_func, *E_params):
    """Compute quadrature approximation of the integral."""
    a = 1.0 / (1.0 + z * nodes)
    E_vals = E_func(a, *E_params)
    return np.sum(weights / E_vals)


def _generate_node_families():
    """Generate diverse node families for testing."""
    families = []
    
    # Uniform
    families.append(np.linspace(0.05, 0.95, 8))
    
    # Beta-distributed nodes
    from scipy.stats import beta as beta_dist
    for alpha, b in [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3), (0.2, 0.8), (0.8, 0.2),
                      (0.5, 1.5), (1.5, 0.5), (0.4, 0.6), (0.6, 0.4), (0.3, 1.0)]:
        qs = np.linspace(0.005, 0.995, 8)
        nodes = beta_dist.ppf(qs, alpha, b)
        nodes = np.clip(nodes, 0.001, 0.999)
        nodes = np.sort(nodes)
        if np.all(np.diff(nodes) > 1e-6):
            families.append(nodes)
    
    # Power-law spaced
    for p in [0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0]:
        t = np.linspace(0, 1, 8)
        nodes = 0.005 + 0.99 * t ** p
        families.append(nodes)
    
    # Log-spaced
    log_nodes = np.linspace(np.log(0.005), np.log(0.995), 8)
    families.append(np.exp(log_nodes))
    
    # Gauss-Legendre mapped to [0.005, 0.995]
    nodes_gl, _ = np.polynomial.legendre.leggauss(8)
    nodes_gl = 0.5 + 0.495 * nodes_gl
    nodes_gl = np.sort(nodes_gl)
    families.append(nodes_gl)
    
    # Chebyshev nodes mapped
    k = np.arange(8)
    nodes_cheb = 0.5 + 0.495 * np.cos(np.pi * (2*k + 1) / 16)
    nodes_cheb = np.sort(nodes_cheb)
    families.append(nodes_cheb)
    
    # Clustered near 0 (various)
    for exp in [2, 3, 4, 5]:
        t = np.linspace(0, 1, 8)
        nodes = 0.005 + 0.99 * t ** exp
        families.append(nodes)
    
    # Clustered near 1
    for exp in [2, 3, 4]:
        t = np.linspace(0, 1, 8)
        nodes = 0.005 + 0.99 * (1 - (1-t) ** exp)
        families.append(nodes)
    
    # Hand-crafted
    families.append(np.array([0.01, 0.03, 0.07, 0.14, 0.25, 0.4, 0.6, 0.85]))
    families.append(np.array([0.005, 0.02, 0.05, 0.1, 0.2, 0.35, 0.55, 0.8]))
    families.append(np.array([0.005, 0.015, 0.04, 0.08, 0.15, 0.28, 0.5, 0.8]))
    families.append(np.array([0.01, 0.04, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9]))
    families.append(np.array([0.02, 0.05, 0.1, 0.18, 0.3, 0.45, 0.65, 0.85]))
    families.append(np.array([0.005, 0.03, 0.08, 0.15, 0.28, 0.45, 0.65, 0.85]))
    families.append(np.array([0.01, 0.05, 0.12, 0.22, 0.35, 0.5, 0.7, 0.9]))
    
    # Symmetric
    for spread in [0.3, 0.4, 0.45]:
        offsets = np.linspace(-spread, spread, 8)
        nodes = 0.5 + offsets
        if nodes[0] > 0.001 and nodes[-1] < 0.999:
            families.append(nodes)
    
    # Random structured
    rng = np.random.RandomState(42)
    for _ in range(30):
        x = np.sort(rng.uniform(0.005, 0.995, 8))
        families.append(x)
    
    return families


def _fit_weights(nodes, z_fit, E_funcs, E_params_list):
    """Fit weights via NNLS to match exact integrals across test cosmologies."""
    num_cosmo = len(E_funcs)
    A = np.zeros((num_cosmo, 8))
    b = np.zeros(num_cosmo)
    
    for j, (E_func, E_params) in enumerate(zip(E_funcs, E_params_list)):
        a = 1.0 / (1.0 + z_fit * nodes)
        E_vals = E_func(a, *E_params)
        A[j, :] = 1.0 / E_vals
        b[j] = _exact_integral(z_fit, E_func, *E_params)
    
    # NNLS for non-negative weights
    w, _ = nnls(A, b)
    w = np.maximum(w, 1e-15)
    w_sum = w.sum()
    if w_sum < 1e-15:
        w = np.full(8, 0.125)
    else:
        w = w / w_sum
    
    return w


def _worst_case_error(nodes, weights, z_test_list, E_funcs, E_params_list):
    """Compute worst-case relative error across all test cosmologies and redshifts."""
    max_err = 0.0
    for z in z_test_list:
        for E_func, E_params in zip(E_funcs, E_params_list):
            quad_val = _quad_value(z, nodes, weights, E_func, *E_params)
            exact_val = _exact_integral(z, E_func, *E_params)
            if abs(exact_val) > 1e-15:
                rel_err = abs(quad_val - exact_val) / abs(exact_val)
                max_err = max(max_err, rel_err)
    return max_err


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s margin before 90s kill
    
    # Fitting cosmologies (diverse)
    E_funcs = [
        _E_cpl, _E_cpl, _E_cpl, _E_cpl,
        _E_lcdm, _E_lcdm,
        _E_const_w, _E_const_w,
        _E_exotic, _E_exotic
    ]
    E_params_list = [
        (-0.5, 0.5), (-0.9, 0.3), (-1.0, 0.0), (-0.7, 0.5),
        (0.3, 0.7), (0.2, 0.8),
        (-0.5, 0.5), (-1.0, 0.0),
        (0.5, 2.0), (0.3, 2.5)
    ]
    
    z_fit = 0.5
    z_test_list = [0.3, 0.5, 1.0]
    
    # Generate and evaluate node families
    node_families = _generate_node_families()
    
    best_nodes = None
    best_weights = None
    best_err = np.inf
    
    for family in node_families:
        if time.time() > deadline - 15.0:
            break
        
        nodes = np.array(family, dtype=float)
        if len(nodes) != 8:
            continue
        if not np.all(np.diff(nodes) > 1e-6):
            continue
        if nodes[0] <= 0 or nodes[-1] >= 1:
            continue
        
        weights = _fit_weights(nodes, z_fit, E_funcs, E_params_list)
        if not np.all(weights > 0):
            continue
        weights = weights / weights.sum()
        
        err = _worst_case_error(nodes, weights, z_test_list, E_funcs, E_params_list)
        
        if err < best_err:
            best_err = err
            best_nodes = nodes.copy()
            best_weights = weights.copy()
    
    # Local refinement of best node set
    if best_nodes is not None and time.time() < deadline - 10.0:
        rng = np.random.RandomState(123)
        for _ in range(100):
            if time.time() > deadline - 8.0:
                break
            
            # Random perturbation with decreasing scale
            scale = 0.01 + 0.02 * np.random.rand()
            pert = rng.normal(0, scale, 8)
            new_nodes = best_nodes + pert
            new_nodes = np.clip(new_nodes, 0.001, 0.999)
            new_nodes = np.sort(new_nodes)
            
            if not np.all(np.diff(new_nodes) > 1e-6):
                continue
            
            weights = _fit_weights(new_nodes, z_fit, E_funcs, E_params_list)
            if not np.all(weights > 0):
                continue
            weights = weights / weights.sum()
            
            err = _worst_case_error(new_nodes, weights, z_test_list, E_funcs, E_params_list)
            if err < best_err:
                best_err = err
                best_nodes = new_nodes.copy()
                best_weights = weights.copy()
    
    # Fallback
    if best_nodes is None:
        best_nodes = (np.arange(8) + 0.5) / 8.0
        best_weights = np.full(8, 0.125)
    
    # Final constraint enforcement
    best_nodes = np.clip(best_nodes, 0.001, 0.999)
    best_nodes = np.sort(best_nodes)
    best_weights = np.maximum(best_weights, 1e-15)
    best_weights = best_weights / best_weights.sum()
    
    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
