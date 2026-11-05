"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _E_squared(a, Om, w0, wa):
    """Vectorized E^2(a) for CPL model."""
    Ode = 1.0 - Om
    if Ode < 0:
        Ode = 0.0
    term1 = Om / (a**3)
    term2 = Ode * (a**(-3.0 * w0)) * np.exp(-3.0 * wa * (1.0 - a))
    return np.maximum(term1 + term2, 1e-30)

def _quadrature_value(nodes, weights, z, Om, w0, wa):
    """Compute quadrature approximation for luminosity distance."""
    a_vals = 1.0 / (1.0 + z * nodes)
    E2 = _E_squared(a_vals, Om, w0, wa)
    E = np.sqrt(E2)
    return (1.0 + z) * z * np.dot(weights, 1.0 / E)

def _reference_value_fast(z, Om, w0, wa, n_ref=2000):
    """High-accuracy reference using vectorized trapezoidal rule on t in [0,1]."""
    if z < 1e-12:
        return 0.0
    t_grid = np.linspace(0, 1, n_ref)
    a_vals = 1.0 / (1.0 + z * t_grid)
    E2 = _E_squared(a_vals, Om, w0, wa)
    E = np.sqrt(E2)
    f = 1.0 / E
    integral_val = np.trapezoid(f, t_grid)
    return (1.0 + z) * z * integral_val

def _worst_case_error(nodes, weights, portfolio, z_max=2.0, n_z=50):
    """Compute worst relative error across portfolio with soft-max smoothing."""
    errors = []
    z_grid = np.linspace(0.1, z_max, n_z)
    for Om, w0, wa in portfolio:
        for z in z_grid:
            quad_val = _quadrature_value(nodes, weights, z, Om, w0, wa)
            ref_val = _reference_value_fast(z, Om, w0, wa, n_ref=1500)
            if abs(ref_val) > 1e-10:
                rel_err = abs(quad_val - ref_val) / abs(ref_val) * 100.0
                errors.append(rel_err)
    if not errors:
        return 1e10
    errors = np.array(errors)
    beta = 50.0
    max_err = np.max(errors)
    clipped = np.clip(errors - max_err, -50.0, 50.0)
    log_sum_exp = np.log(np.sum(np.exp(beta * clipped)))
    return max_err + (1.0 / beta) * log_sum_exp

def _build_moment_basis(n_basis=8, n_ref=500):
    """
    Build a set of basis functions that capture the worst-case behavior of 1/E(a(t)).
    
    Uses SVD on collected integrand samples to find dominant modes.
    """
    portfolio = [
        (0.30, -1.0, 0.0),
        (0.30, -1.0, 0.3),
        (0.30, -1.0, -0.3),
        (0.27, -1.2, 0.5),
        (0.27, -0.8, -0.5),
        (0.31, -1.1, 0.2),
        (0.31, -0.9, -0.2),
    ]
    
    z_list = [0.5, 1.0, 1.5, 2.0]
    t_grid = np.linspace(0, 1, n_ref)
    all_integrands = []
    
    for Om, w0, wa in portfolio:
        for z in z_list:
            a_vals = 1.0 / (1.0 + z * t_grid)
            E2 = _E_squared(a_vals, Om, w0, wa)
            E = np.sqrt(E2)
            g = 1.0 / E
            all_integrands.append(g)
    
    all_integrands = np.array(all_integrands)
    
    # SVD to find dominant modes
    U, S, Vt = np.linalg.svd(all_integrands, full_matrices=False)
    
    n_modes = min(n_basis, len(S))
    basis_funcs = []
    from scipy.interpolate import interp1d
    for i in range(n_modes):
        v = Vt[i]
        f_interp = interp1d(t_grid, v, kind='cubic', fill_value='extrapolate')
        basis_funcs.append(f_interp)
    
    return basis_funcs, t_grid, all_integrands, S

def _moment_matching_objective(x, basis_funcs, t_grid, S):
    """
    Objective: minimize weighted sum of squared errors in integrating basis functions.
    """
    nodes = x[:8]
    weights = x[8:]
    
    if np.any(nodes <= 0) or np.any(nodes >= 1):
        return 1e10
    if np.any(np.diff(nodes) <= 0):
        return 1e10
    if np.any(weights <= 0):
        return 1e10
    if abs(np.sum(weights) - 1.0) > 1e-6:
        return 1e10
    
    total_error = 0.0
    
    for i, f_interp in enumerate(basis_funcs):
        quad_val = np.dot(weights, f_interp(nodes))
        exact_val = np.trapezoid(f_interp(t_grid), t_grid)
        
        if abs(exact_val) > 1e-12:
            rel_err = abs(quad_val - exact_val) / abs(exact_val)
        else:
            rel_err = abs(quad_val - exact_val)
        
        # Weight by singular value squared (importance)
        weight = S[i] ** 2 if i < len(S) else 1.0
        total_error += weight * rel_err ** 2
    
    return total_error

def construct():
    t_start = time.time()
    deadline = 85.0
    
    # Phase 1: Build moment basis
    if time.time() - t_start > deadline - 25.0:
        nodes_gl, weights_gl = np.polynomial.legendre.leggauss(8)
        nodes = (nodes_gl + 1.0) / 2.0
        weights = weights_gl / 2.0
        return {"nodes": nodes, "weights": weights}
    
    basis_funcs, t_grid, all_integrands, S = _build_moment_basis(n_basis=8, n_ref=500)
    
    # Phase 2: Initial guess from Gauss-Legendre
    nodes_gl, weights_gl = np.polynomial.legendre.leggauss(8)
    x0 = np.concatenate([(nodes_gl + 1.0) / 2.0, weights_gl / 2.0])
    
    def _objective(x):
        if time.time() - t_start > deadline:
            return 1e10
        return _moment_matching_objective(x, basis_funcs, t_grid, S)
    
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x[8:]) - 1.0},
    ]
    bounds = [(1e-6, 1 - 1e-6)] * 8 + [(1e-8, 1.0)] * 8
    
    best_x = x0.copy()
    best_obj = _objective(x0)
    
    # Phase 3: Optimize using moment matching
    while time.time() - t_start < deadline - 20.0:
        try:
            res = minimize(_objective, best_x, method='SLSQP', bounds=bounds,
                         constraints=constraints, options={'maxiter': 80, 'ftol': 1e-10})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x
        except Exception:
            break
    
    # Phase 4: Validate and fine-tune against actual cosmologies
    if time.time() - t_start < deadline - 5.0:
        portfolio = [
            (0.30, -1.0, 0.0),
            (0.30, -1.0, 0.3),
            (0.30, -1.0, -0.3),
            (0.27, -1.2, 0.5),
            (0.27, -0.8, -0.5),
            (0.31, -1.1, 0.2),
            (0.31, -0.9, -0.2),
        ]
        
        nodes_mm = best_x[:8]
        weights_mm = best_x[8:]
        err_mm = _worst_case_error(nodes_mm, weights_mm, portfolio, z_max=2.0, n_z=30)
        
        nodes_gl2, weights_gl2 = np.polynomial.legendre.leggauss(8)
        nodes_gl2 = (nodes_gl2 + 1.0) / 2.0
        weights_gl2 = weights_gl2 / 2.0
        err_gl = _worst_case_error(nodes_gl2, weights_gl2, portfolio, z_max=2.0, n_z=30)
        
        if err_mm < err_gl:
            best_x = np.concatenate([nodes_mm, weights_mm])
            best_obj = err_mm
        else:
            best_x = np.concatenate([nodes_gl2, weights_gl2])
            best_obj = err_gl
        
        def _final_objective(x):
            if time.time() - t_start > deadline:
                return 1e10
            nodes = x[:8]
            weights = x[8:]
            if np.any(nodes <= 0) or np.any(nodes >= 1):
                return 1e10
            if np.any(np.diff(nodes) <= 0):
                return 1e10
            if np.any(weights <= 0):
                return 1e10
            if abs(np.sum(weights) - 1.0) > 1e-6:
                return 1e10
            return _worst_case_error(nodes, weights, portfolio, z_max=2.0, n_z=40)
        
        try:
            res = minimize(_final_objective, best_x, method='SLSQP', bounds=bounds,
                         constraints=constraints, options={'maxiter': 120, 'ftol': 1e-12})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x
        except Exception:
            pass
    
    nodes = best_x[:8]
    weights = best_x[8:]
    
    weights = np.maximum(weights, 1e-10)
    weights = weights / np.sum(weights)
    nodes = np.sort(nodes)
    for i in range(1, 8):
        if nodes[i] - nodes[i-1] < 1e-10:
            nodes[i] = nodes[i-1] + 1e-10
    nodes = np.clip(nodes, 1e-8, 1 - 1e-8)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
