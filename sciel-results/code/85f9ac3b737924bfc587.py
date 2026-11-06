"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad

_start_time = time.time()
DEADLINE = 80.0  # safety margin before 90s kill

def _deadline_ok():
    return (time.time() - _start_time) < DEADLINE

def _E_of_a(a, Om0, Ok0, Ol0, w0, w1):
    """H(a)/H0 for CPL model."""
    a = np.asarray(a)
    a3 = a ** 3
    a2 = a ** 2
    # w(z) = w0 + w1*z, z = 1/a - 1
    w = w0 + w1 * (1.0 / a - 1.0)
    de = Ol0 * a ** (-3.0 * (1.0 + w))
    return np.sqrt(Om0 / a3 + Ok0 / a2 + de)

def _integrand(t, z, Om0, Ok0, Ol0, w0, w1):
    """1/E(z*t) where a = 1/(1+z*t)"""
    a = 1.0 / (1.0 + z * t)
    return 1.0 / _E_of_a(a, Om0, Ok0, Ol0, w0, w1)

def _exact_integral(z, Om0, Ok0, Ol0, w0, w1):
    """High-precision integral of 1/E(z*t) from 0 to 1."""
    val, _ = quad(lambda t: _integrand(t, z, Om0, Ok0, Ol0, w0, w1), 0, 1, limit=200)
    return val

def _quadrature_value(x, w, z, Om0, Ok0, Ol0, w0, w1):
    """Compute quadrature sum for given nodes x and weights w."""
    a = 1.0 / (1.0 + z * x)
    E_vals = _E_of_a(a, Om0, Ok0, Ol0, w0, w1)
    return np.sum(w / E_vals)

def _rel_error(x, w, z, Om0, Ok0, Ol0, w0, w1):
    exact = _exact_integral(z, Om0, Ok0, Ol0, w0, w1)
    approx = _quadrature_value(x, w, z, Om0, Ok0, Ol0, w0, w1)
    return abs(approx - exact) / abs(exact)

def _objective(params):
    """Optimize log-weights and nodes to minimize max rel error."""
    # params: [x1..x8, logw1..logw8]
    x = params[:8]
    logw = params[8:]
    w = np.exp(logw)
    w = w / np.sum(w)  # normalize to sum to 1
    
    # Check nodes are in (0,1) and increasing
    if np.any(x <= 0) or np.any(x >= 1) or np.any(np.diff(x) <= 0):
        return 1e10
    
    max_err = 0.0
    for z, Om0, Ok0, Ol0, w0, w1 in _test_set:
        if not _deadline_ok():
            return 1e10
        err = _rel_error(x, w, z, Om0, Ok0, Ol0, w0, w1)
        max_err = max(max_err, err)
    return max_err

# Test cosmologies: (z, Om0, Ok0, Ol0, w0, w1)
_test_set = [
    (0.5, 0.3, 0.0, 0.7, -1.0, 0.0),   # LCDM, z=0.5
    (1.0, 0.3, 0.0, 0.7, -1.0, 0.0),   # LCDM, z=1.0
    (2.0, 0.3, 0.0, 0.7, -1.0, 0.0),   # LCDM, z=2.0
    (0.5, 0.3, 0.0, 0.7, -0.9, 0.2),   # CPL, z=0.5
    (1.0, 0.3, 0.0, 0.7, -0.9, 0.2),   # CPL, z=1.0
    (2.0, 0.3, 0.0, 0.7, -0.9, 0.2),   # CPL, z=2.0
    (0.5, 0.3, 0.0, 0.7, -1.1, -0.3),  # CPL, z=0.5
    (1.0, 0.3, 0.0, 0.7, -1.1, -0.3),  # CPL, z=1.0
    (2.0, 0.3, 0.0, 0.7, -1.1, -0.3),  # CPL, z=2.0
    (0.5, 0.3, 0.05, 0.65, -1.0, 0.0), # LCDM+curv, z=0.5
    (1.0, 0.3, 0.05, 0.65, -1.0, 0.0), # LCDM+curv, z=1.0
    (2.0, 0.3, 0.05, 0.65, -1.0, 0.0), # LCDM+curv, z=2.0
    (0.5, 0.3, 0.05, 0.65, -0.8, 0.3), # CPL+curv, z=0.5
    (1.0, 0.3, 0.05, 0.65, -0.8, 0.3), # CPL+curv, z=1.0
    (2.0, 0.3, 0.05, 0.65, -0.8, 0.3), # CPL+curv, z=2.0
    (3.0, 0.3, 0.0, 0.7, -1.0, 0.0),   # LCDM, z=3.0
    (3.0, 0.3, 0.0, 0.7, -0.9, 0.2),   # CPL, z=3.0
    (0.3, 0.3, 0.0, 0.7, -1.0, 0.0),   # LCDM, z=0.3
    (0.3, 0.3, 0.0, 0.7, -1.1, -0.3),  # CPL, z=0.3
]

def construct():
    global _test_set
    n = 8
    
    # Initial guess: Gauss-Legendre-like nodes
    x0 = np.linspace(0.05, 0.95, n)
    w0 = np.ones(n) / n
    logw0 = np.log(w0)
    params0 = np.concatenate([x0, logw0])
    
    best_params = params0.copy()
    best_obj = 1e10
    
    # Try multiple restarts
    for restart in range(5):
        if not _deadline_ok():
            break
        
        if restart == 0:
            p = params0.copy()
        else:
            p = params0 + np.random.randn(16) * 0.05
        
        result = minimize(
            _objective, p, method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x.copy()
        
        if not _deadline_ok():
            break
    
    # Extract and normalize
    x = best_params[:8]
    logw = best_params[8:]
    w = np.exp(logw)
    w = w / np.sum(w)
    
    # Ensure nodes are strictly in (0,1) and increasing
    x = np.sort(x)
    x = np.clip(x, 1e-6, 1 - 1e-6)
    # Ensure strict increase
    for i in range(1, n):
        if x[i] <= x[i-1]:
            x[i] = x[i-1] + 1e-6
    
    # Final normalization
    w = w / np.sum(w)
    
    return {"nodes": x, "weights": w}
# EVOLVE-BLOCK-END
