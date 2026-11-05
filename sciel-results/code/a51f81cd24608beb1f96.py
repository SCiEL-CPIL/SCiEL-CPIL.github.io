"""Cosmology: universal luminosity-distance quadrature.
STRATEGY: Direct Global Minimax Optimization with Smart Initialization.
This approach abandons the spectral residual method in favor of directly minimizing
the worst-case relative error using Nelder-Mead optimization. We generate multiple
initial guesses from different quadrature families (Gauss-Legendre, Gauss-Chebyshev,
Clenshaw-Curtis, and randomized) and run Nelder-Mead on each. This allows the
optimizer to find nodes that are specifically tailored to the 1/E(a) integrand
across a diverse library of CPL cosmologies, avoiding the potential local minima
traps of spectral approximation methods.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

START = time.time()
DEADLINE = 90.0
MARGIN = 5.0

def _fallback():
    n = 8
    return {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}

def _E_cpl(a, w0, wa):
    """CPL E(a) for flat universe."""
    a = np.clip(a, 1e-8, 1.0)
    Omega_m = 0.3
    Omega_L0 = 0.7
    exp_term = np.exp(3.0 * wa * (a - 1.0))
    pow_term = a ** (-3.0 * (1.0 + w0 + wa))
    Omega_L = Omega_L0 * pow_term * exp_term
    E2 = a ** (-3.0) + Omega_L
    return np.sqrt(E2)

def _integrate(z, nodes, weights, Efun):
    """Compute the quadrature approximation."""
    a = 1.0 / (1.0 + z * nodes)
    E = Efun(a)
    return (1.0 + z) * z * np.sum(weights / E)

def _ref_integrate(z, Efun):
    """High-accuracy reference integration using Gauss-Legendre."""
    n = 64
    xi, wi = np.polynomial.legendre.leggauss(n)
    u = 0.5 * (xi + 1.0)
    a = 1.0 / (1.0 + z * u)
    E = Efun(a)
    val = z * np.sum(wi / E)
    return (1.0 + z) * val

def _build_library():
    """Build a diverse library of (w0, wa) pairs for validation."""
    # Wider range to capture potential validation set
    w0_vals = np.linspace(-2.5, 1.5, 8)
    wa_vals = np.linspace(-2.5, 2.5, 8)
    lib = []
    for w0 in w0_vals:
        for wa in wa_vals:
            lib.append((w0, wa))
    return lib

def _get_error(nodes, weights, zs, lib):
    """Compute worst-case relative error for the given nodes/weights."""
    max_err = 0.0
    for w0, wa in lib:
        Efun = lambda a, w0=w0, wa=wa: _E_cpl(a, w0, wa)
        for z in zs:
            app = _integrate(z, nodes, weights, Efun)
            ref = _ref_integrate(z, Efun)
            if ref == 0:
                continue
            err = abs(app - ref) / ref
            if err > max_err:
                max_err = err
    return max_err

def _generate_initial_guesses(n):
    """Generate multiple initial guesses for nodes and weights."""
    guesses = []
    
    # 1. Gauss-Legendre
    xi, wi = np.polynomial.legendre.leggauss(n)
    nodes_gl = 0.5 * (xi + 1.0)
    weights_gl = 0.5 * wi
    weights_gl = weights_gl / np.sum(weights_gl)
    guesses.append((nodes_gl, weights_gl))
    
    # 2. Gauss-Chebyshev
    theta = (np.arange(n) + 0.5) * np.pi / n
    nodes_gc = 0.5 * (1 - np.cos(theta))
    weights_gc = np.full(n, np.pi / n)
    weights_gc = weights_gc / np.sum(weights_gc)
    guesses.append((nodes_gc, weights_gc))
    
    # 3. Clenshaw-Curtis
    nodes_cc = 0.5 * (1 - np.cos(np.pi * np.arange(n) / (n - 1)))
    # Approximate weights for CC
    weights_cc = np.zeros(n)
    weights_cc[0] = 1.0 / (n - 1)
    weights_cc[-1] = 1.0 / (n - 1)
    for k in range(1, n - 1):
        weights_cc[k] = 2.0 / (n - 1)
    weights_cc = weights_cc / np.sum(weights_cc)
    guesses.append((nodes_cc, weights_cc))
    
    # 4. Randomized uniform
    rng = np.random.default_rng(42)
    for _ in range(5):
        nodes_rand = np.sort(rng.uniform(0.1, 0.9, n))
        weights_rand = np.full(n, 1.0 / n)
        guesses.append((nodes_rand, weights_rand))
        
    return guesses

def construct():
    fallback = _fallback()
    n = 8
    
    # Use a moderate set of z values for speed
    zs = np.linspace(0.1, 3.0, 15)
    lib = _build_library()
    
    best_nodes = None
    best_weights = None
    best_err = np.inf
    
    def error_obj(x):
        nodes_x = x[:n]
        weights_x = x[n:]
        
        # Constraints: nodes in [0,1], strictly increasing
        if nodes_x[0] < 1e-6 or nodes_x[-1] > 1 - 1e-6:
            return 1e10
        if np.any(np.diff(nodes_x) < 1e-6):
            return 1e10
        if np.any(weights_x < 1e-12):
            return 1e10
            
        weights_norm = weights_x / np.sum(weights_x)
        return _get_error(nodes_x, weights_norm, zs, lib)
        
    # Try each initial guess
    for nodes_init, weights_init in _generate_initial_guesses(n):
        if time.time() > DEADLINE - MARGIN:
            break
            
        x0 = np.concatenate([nodes_init, weights_init])
        
        # Use Nelder-Mead for direct minimax optimization
        res = minimize(error_obj, x0, method='Nelder-Mead',
                       options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if res.fun < best_err:
            new_nodes = res.x[:n]
            new_weights = res.x[n:]
            
            # Check validity
            if np.all(np.diff(new_nodes) > 1e-6) and np.all(new_weights > 1e-12):
                new_weights = new_weights / np.sum(new_weights)
                best_nodes = new_nodes
                best_weights = new_weights
                best_err = res.fun

    # Final validation and return
    if best_nodes is not None and np.all(np.diff(best_nodes) > 0) and np.all(best_weights > 0):
        best_weights = best_weights / np.sum(best_weights)
        return {"nodes": best_nodes, "weights": best_weights}
    
    return fallback
# EVOLVE-BLOCK-END
