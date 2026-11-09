"""Direct construction using the Cohn-Goncalves Laguerre polynomial root structure."""
import numpy as np
import time
from numpy.polynomial.laguerre import laggauss
from scipy.special import genlaguerre, eval_genlaguerre
from scipy.optimize import minimize, differential_evolution

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = 600.0
_SAFETY_MARGIN = 30.0

def _deadline_reached():
    return (time.time() - _START_TIME) >= (_DEADLINE - _SAFETY_MARGIN)

def construct():
    """
    Construct the 12 positive double-root locations for the Cohn-Goncalves
    Laguerre polynomial construction with m=12, n=1.
    
    Strategy: Use the exact zeros of the generalized Laguerre polynomial L_{12}^{(1)}(x)
    as the root locations, then refine with numerical optimization of the actual C4 bound
    computed via Gauss-Laguerre quadrature.
    """
    m = 12
    n = 1
    
    # Get initial roots: zeros of L_{12}^{(1)}(x)
    degree = m  # 12
    alpha = 2 * n - 1  # = 1
    
    try:
        L = genlaguerre(degree, alpha)
        roots = L.roots()
        roots = roots[np.real(roots) > 0]
        roots = np.real(roots)
        roots = np.sort(roots)
        
        if len(roots) != m:
            if len(roots) > m:
                roots = roots[:m]
            else:
                roots = _get_fallback_roots(m, n)
    except Exception:
        roots = _get_fallback_roots(m, n)
    
    # Verify roots are positive
    if not np.all(roots > 0):
        roots = np.maximum(roots, 1e-10)
    
    # Try to refine with numerical optimization
    if not _deadline_reached():
        try:
            best_roots, best_score = _optimize_roots(roots, m, n)
            return [float(x) for x in best_roots]
        except Exception:
            pass
    
    return [float(x) for x in roots]

def _get_fallback_roots(m, n):
    """Fallback: use Laguerre-Gauss quadrature nodes as proxy."""
    try:
        x, w = laggauss(m, alpha=2*n-1)
        return np.sort(x)
    except Exception:
        return np.array([4.0 * (i + 1) for i in range(m)], dtype=float)

def _compute_c4_bound(roots, m=12, n=1, num_quad=200):
    """
    Compute the C4 uncertainty upper bound for given root locations.
    
    The Cohn-Goncalves construction gives a sign uncertainty bound.
    We compute it using Gauss-Laguerre quadrature.
    
    For the Fourier sign uncertainty, the bound involves the integral of
    |f(x)|^2 weighted appropriately. The refined construction with double
    roots at positions r_1, ..., r_m gives an upper bound on C4.
    
    The key formula: the bound is related to the norm of the polynomial
    constructed from these roots in the Laguerre basis.
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != m or np.any(roots <= 0):
        return np.inf
    
    # For the Cohn-Goncalves construction, the uncertainty bound C4
    # is computed from the Laguerre polynomial combination.
    # 
    # The construction: P(x) = product of (x - r_i)^2 for i=1..m
    # This is a polynomial of degree 2m = 24.
    # 
    # The C4 bound relates to the L2 norm of this polynomial against
    # the Laguerre weight, normalized appropriately.
    
    # Compute the polynomial P(x) = prod (x - r_i)^2
    # Using the roots to build the polynomial
    coeffs = np.ones(1)
    for r in roots:
        # Multiply by (x - r)^2 = x^2 - 2r*x + r^2
        new_coeffs = np.zeros(len(coeffs) + 2)
        new_coeffs[0] = coeffs[0]
        new_coeffs[1] = coeffs[1] - 2*r*coeffs[0] if len(coeffs) > 1 else -2*r*coeffs[0]
        for j in range(2, len(coeffs)):
            new_coeffs[j] = coeffs[j] - 2*r*coeffs[j-1] + r*r*coeffs[j-2]
        if len(coeffs) >= 2:
            new_coeffs[-1] = r*r*coeffs[-1]
        coeffs = new_coeffs
    
    # Now compute the L2 norm with Laguerre weight: integral_0^inf |P(x)|^2 * x^(2n-1) * e^{-x} dx
    # For n=1: weight is x^1 * e^{-x}
    # Use Gauss-Laguerre quadrature
    try:
        x_quad, w_quad = laggauss(num_quad)
        # Evaluate |P(x)|^2 at quadrature points
        P_vals = np.polyval(coeffs, x_quad)
        weight_vals = x_quad * np.exp(-x_quad)  # x^(2n-1) * e^{-x} for n=1
        integral = np.sum(w_quad * P_vals**2 * weight_vals)
        
        # The C4 bound is proportional to this integral, normalized
        # by appropriate factors. The exact normalization depends on the
        # specific construction, but the key point is minimizing this integral.
        c4_bound = integral
        
        # Also compute a normalization factor
        # The norm of the constant polynomial 1 in this weighted space:
        # integral_0^inf 1^2 * x * e^{-x} dx = Gamma(2) = 1! = 1
        # So the bound is just the integral above (for n=1)
        
        return c4_bound
    except Exception:
        return np.inf

def _compute_c4_bound_v2(roots, m=12, n=1, num_quad=150):
    """
    Alternative computation of the C4 bound using a more direct approach.
    
    For the Cohn-Goncalves construction with m double roots, the sign
    uncertainty bound C4 can be computed as follows:
    
    The polynomial P(x) = prod_{i=1}^m (x - r_i)^2 has degree 2m.
    In the Laguerre basis with weight x^{2n-1}e^{-x}, the squared norm
    gives the bound.
    
    For n=1, weight is x*e^{-x}, and the Laguerre polynomials L_k(x)
    are orthogonal with <L_i, L_j> = delta_{ij} * Gamma(i+1)/i! ... 
    Actually <L_i, L_j> = delta_{ij} * Gamma(i+2)/i! for weight x*e^{-x}.
    
    Let me use a simpler but effective approach: compute the integral directly.
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != m or np.any(roots <= 0):
        return np.inf
    
    try:
        # Build P(x) = prod (x - r_i)^2 using logarithms for stability
        # Actually, let's just evaluate directly at quadrature points
        
        x_quad, w_quad = laggauss(num_quad)
        
        # P(x) = prod (x - r_i)^2
        P_vals = np.ones_like(x_quad)
        for r in roots:
            P_vals *= (x_quad - r)**2
        
        weight_vals = x_quad * np.exp(-x_quad)
        integral = np.sum(w_quad * P_vals**2 * weight_vals)
        
        return integral
    except Exception:
        return np.inf

def _optimize_roots(initial_roots, m, n):
    """
    Optimize the root locations to minimize the C4 bound.
    """
    initial_roots = np.asarray(initial_roots, dtype=float)
    
    # Use differential evolution for global optimization
    # Bounds: each root should be positive, roughly in [0.1, 50]
    bounds = [(0.1, 50.0)] * m
    
    def objective(x):
        if _deadline_reached():
            return np.inf
        x = np.sort(np.abs(x))  # Ensure positive and sorted
        if np.any(x <= 0):
            return np.inf
        return _compute_c4_bound_v2(x, m, n)
    
    # First try a local refinement from the initial roots
    try:
        result = minimize(
            lambda x: _compute_c4_bound_v2(np.sort(np.abs(x)), m, n),
            initial_roots,
            method='L-BFGS-B',
            bounds=[(0.01, 100.0)] * m,
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        best_roots = np.sort(np.abs(result.x))
        best_score = result.fun
        
        if _deadline_reached():
            return best_roots, best_score
    except Exception:
        best_roots = initial_roots.copy()
        best_score = _compute_c4_bound_v2(initial_roots, m, n)
    
    # Try differential evolution if time permits
    if not _deadline_reached():
        try:
            de_result = differential_evolution(
                objective,
                bounds,
                maxiter=200,
                popsize=15,
                tol=1e-10,
                seed=42,
                polish=True,
                init='sobol'
            )
            if de_result.fun < best_score:
                best_roots = np.sort(np.abs(de_result.x))
                best_score = de_result.fun
        except Exception:
            pass
    
    return best_roots, best_score

# EVOLVE-BLOCK-END
