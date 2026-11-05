"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _make_E(Omega_m, w0, wa):
    """Create E(a) function for a CPL cosmology."""
    def E(a):
        # E^2(a) = Omega_m/a^3 + Omega_Lambda * a^(-3(1+w0+wa))
        # Omega_Lambda = 1 - Omega_m (flat)
        Omega_L = 1.0 - Omega_m
        term_m = Omega_m / (a**3)
        term_L = Omega_L * np.exp(-3.0*(1.0+w0+wa)*np.log(a))
        return np.sqrt(term_m + term_L)
    return E

def _integrand_values(x_nodes, z, E_func):
    """Compute 1/E(a_i) where a_i = 1/(1+z*x_i)"""
    a_vals = 1.0 / (1.0 + z * x_nodes)
    return 1.0 / E_func(a_vals)

def _true_integral(z, E_func, n_quad=10000):
    """Compute the true integral using adaptive quadrature."""
    from scipy.integrate import quad
    def integrand(x):
        a = 1.0 / (1.0 + z * x)
        return 1.0 / E_func(a)
    result, _ = quad(integrand, 0, 1, limit=100)
    return result

def _quadrature_estimate(x_nodes, weights, z, E_func):
    """Compute quadrature estimate of the integral."""
    vals = _integrand_values(x_nodes, z, E_func)
    return np.dot(weights, vals)

def _relative_error(x_nodes, weights, z, E_func):
    """Compute relative error for a given z and model."""
    true_val = _true_integral(z, E_func)
    quad_val = _quadrature_estimate(x_nodes, weights, z, E_func)
    return abs(quad_val - true_val) / abs(true_val)

def _worst_case_error(x_nodes, weights, models, z_values):
    """Compute worst-case relative error across all models and z values."""
    max_err = 0.0
    for model in models:
        Omega_m, w0, wa = model
        E_func = _make_E(Omega_m, w0, wa)
        for z in z_values:
            err = _relative_error(x_nodes, weights, z, E_func)
            if err > max_err:
                max_err = err
    return max_err

def _objective(params):
    """Objective function for optimization."""
    # params: [x1, x2, ..., x8, w1, w2, ..., w8]
    x = params[:8]
    w = params[8:16]
    
    # Enforce positive weights
    if np.any(w <= 0):
        return 1e10
    
    # Normalize weights
    w_norm = w / np.sum(w)
    
    # Enforce strictly increasing nodes in (0,1)
    if x[0] <= 0 or x[-1] >= 1:
        return 1e10
    if np.any(np.diff(x) <= 0):
        return 1e10
    
    return _worst_case_error(x, w_norm, _MODELS, _Z_VALUES)

# Pre-define models and z values for optimization
_MODELS = [
    (0.3, -1.0, 0.0),    # LCDM
    (0.3, -1.0, 0.5),    # CPL with positive wa
    (0.3, -1.0, -0.5),   # CPL with negative wa
    (0.3, -0.8, 0.0),    # Quintessence-like
    (0.3, -1.2, 0.0),    # Phantom-like
    (0.25, -1.0, 0.3),
    (0.35, -1.0, -0.3),
    (0.3, -0.9, 0.2),
    (0.3, -1.1, -0.2),
]

_Z_VALUES = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s
    
    # Start with midpoint rule
    n = 8
    x0 = (np.arange(n) + 0.5) / n
    w0 = np.full(n, 1.0 / n)
    
    best_params = np.concatenate([x0, w0])
    best_err = _worst_case_error(x0, w0, _MODELS, _Z_VALUES)
    
    # Try optimization with multiple starting points
    random_starts = [
        # Midpoint
        np.concatenate([(np.arange(n) + 0.5) / n, np.full(n, 1.0/n)]),
        # Gauss-Legendre nodes
        np.concatenate([np.array([0.05764737, 0.20119406, 0.34869883, 0.49999999, 0.65130117, 0.79880594, 0.94235263, 0.99999999])[:8], np.full(n, 1.0/n)]),
        # Uniform with slight perturbation
        np.concatenate([np.linspace(0.05, 0.95, n), np.full(n, 1.0/n)]),
    ]
    
    # Add some random starts
    rng = np.random.RandomState(42)
    for _ in range(3):
        x_rand = np.sort(rng.uniform(0.02, 0.98, n))
        w_rand = rng.dirichlet(np.ones(n))
        random_starts.append(np.concatenate([x_rand, w_rand]))
    
    for i, init in enumerate(random_starts):
        if time.time() > deadline - 10.0:
            break
        
        try:
            result = minimize(
                _objective,
                init,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if result.fun < best_err:
                best_err = result.fun
                best_params = result.x
        except Exception:
            continue
    
    # Extract best nodes and weights
    x_best = best_params[:8]
    w_best = best_params[8:16]
    
    # Ensure weights are positive and sum to 1
    w_best = np.maximum(w_best, 1e-15)
    w_best = w_best / np.sum(w_best)
    
    # Ensure nodes are strictly increasing in (0,1)
    x_best = np.clip(x_best, 1e-10, 1.0 - 1e-10)
    # Ensure strictly increasing
    for i in range(1, 8):
        if x_best[i] <= x_best[i-1]:
            x_best[i] = x_best[i-1] + 1e-10
    x_best = np.clip(x_best, 1e-10, 1.0 - 1e-10)
    
    # Final normalization
    w_best = w_best / np.sum(w_best)
    
    return {"nodes": x_best, "weights": w_best}
# EVOLVE-BLOCK-END
