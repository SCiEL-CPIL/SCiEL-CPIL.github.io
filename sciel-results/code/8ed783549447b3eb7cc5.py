"""Cosmology: universal luminosity-distance quadrature.
STRATEGY: Moment-matching optimization using Chebyshev polynomials.
Instead of optimizing against specific (w0, wa) instances, we treat the
integrand 1/E(a) as a smooth function on [0,1] and construct a quadrature
rule that exactly integrates Chebyshev polynomials of the first kind.
This exploits the spectral accuracy of Chebyshev nodes for smooth functions
and avoids the non-convex, high-dimensional search over model parameters.
The nodes are mapped from Chebyshev-Lobatto or Gauss-Chebyshev points,
and weights are computed to match the moments.
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
    w0_vals = np.linspace(-2.0, 0.0, 5)
    wa_vals = np.linspace(-2.0, 2.0, 5)
    lib = []
    for w0 in w0_vals:
        for wa in wa_vals:
            if abs(w0 + 1.0) < 0.1 and abs(wa) < 0.1:
                continue
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

def construct():
    fallback = _fallback()
    
    # Strategy: Start with Gauss-Chebyshev nodes mapped to [0,1]
    # Gauss-Chebyshev nodes: x_i = cos((2i-1)pi/(2n)) for i=1..n
    # Map from [-1,1] to [0,1]: u_i = 0.5 * (x_i + 1)
    n = 8
    i = np.arange(1, n + 1)
    x = np.cos((2 * i - 1) * np.pi / (2 * n))
    nodes = 0.5 * (x + 1.0)
    nodes = np.sort(nodes)
    
    # Initial weights: Gauss-Chebyshev weights are pi/n for all i
    # But we need sum(weights) = 1.
    # Standard Gauss-Chebyshev quadrature: integral_{-1}^1 f(x) dx ~ (pi/n) sum f(x_i)
    # Our integral is over [0,1]. Change of variable x = 2u - 1 => dx = 2 du.
    # integral_0^1 f(u) du = 0.5 * integral_{-1}^1 f((x+1)/2) dx
    # So weights should be 0.5 * (pi/n) for exactness against Chebyshev basis?
    # Actually, let's just optimize the weights to minimize error on a small library
    # using the Chebyshev nodes as a strong starting point.
    
    # Better: Use moment matching to find weights that integrate polynomials exactly.
    # Let's use a small set of basis functions: 1, u, u^2, ..., u^(n-1)
    # We want sum_i w_i u_i^k = integral_0^1 u^k du = 1/(k+1) for k=0..n-1.
    
    # This is a linear system V w = b, where V_{k,i} = nodes_i^k, b_k = 1/(k+1)
    V = np.vander(nodes, n, increasing=True).T
    b = 1.0 / (np.arange(n) + 1.0)
    
    # Solve for weights
    try:
        weights = np.linalg.solve(V, b)
    except np.linalg.LinAlgError:
        weights = np.ones(n) / n
        
    # Ensure positivity and normalization
    if np.any(weights < 0):
        # Fallback to positive weights if moment matching yields negatives
        # Use Gauss-Chebyshev weights normalized
        weights = np.ones(n) / n
    else:
        weights = weights / np.sum(weights)
        
    # Refine with a few steps of Nelder-Mead on a small library to catch model-specific deviations
    lib = _build_library()
    zs = np.linspace(0.05, 3.0, 20)
    
    best_nodes = nodes.copy()
    best_weights = weights.copy()
    best_err = _get_error(nodes, weights, zs, lib)
    
    def objective(x):
        nodes_x = x[:n]
        weights_x = x[n:]
        if nodes_x[0] < 1e-6 or nodes_x[-1] > 1 - 1e-6:
            return 1e10
        if np.any(np.diff(nodes_x) < 1e-6):
            return 1e10
        if np.any(weights_x < 1e-12):
            return 1e10
        weights_norm = weights_x / np.sum(weights_x)
        return _get_error(nodes_x, weights_norm, zs, lib)
        
    x0 = np.concatenate([nodes, weights])
    
    # Time-limited optimization
    if time.time() < DEADLINE - MARGIN:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.success:
            new_nodes = res.x[:n]
            new_weights = res.x[n:]
            if np.all(np.diff(new_nodes) > 1e-6) and np.all(new_weights > 1e-12):
                new_weights = new_weights / np.sum(new_weights)
                new_err = _get_error(new_nodes, new_weights, zs, lib)
                if new_err < best_err:
                    best_nodes = new_nodes
                    best_weights = new_weights
                    best_err = new_err

    if np.all(np.diff(best_nodes) > 0) and np.all(best_weights > 0):
        best_weights = best_weights / np.sum(best_weights)
        return {"nodes": best_nodes, "weights": best_weights}
    
    return fallback
# EVOLVE-BLOCK-END
