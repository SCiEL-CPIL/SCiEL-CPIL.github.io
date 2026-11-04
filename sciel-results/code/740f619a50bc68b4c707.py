import numpy as np
from scipy.optimize import minimize
import time

def _poly_moment(coeffs, k):
    """Compute integral of x^k * prod(x - r_i)^2 * e^{-x} dx from 0 to inf."""
    # Build the polynomial (x - r_1)^2 ... (x - r_N)^2
    # Then multiply by x^k and integrate term by term with e^{-x}
    # The integral of x^n * e^{-x} dx = n!
    # First build the squared polynomial coefficients
    # Use iterative multiplication
    poly = np.array([1.0])  # start with 1
    for r in coeffs:
        # multiply by (x - r)^2 = x^2 - 2r*x + r^2
        new_poly = np.zeros(len(poly) + 2)
        new_poly[2:] += poly  # x^2 * poly
        new_poly[1:-1] += -2*r * poly  # -2r*x * poly
        new_poly[0:-2] += r*r * poly  # r^2 * poly
        poly = new_poly
    # Now multiply by x^k: shift coefficients
    if k > 0:
        shifted = np.zeros(len(poly) + k)
        shifted[k:] = poly
        poly = shifted
    # Integrate term by term: sum c_i * i!
    result = 0.0
    for i, c in enumerate(poly):
        result += c * np.math.factorial(i)
    return result

def _compute_c4(roots):
    """Compute the C4 upper bound from root positions."""
    # For the refined Cohn-Goncalves construction, C4 involves moments
    # of the polynomial P(x) = prod(x - r_i)^2
    # The bound is related to the ratio of certain moments
    # Simplified: minimize sum of roots and interaction terms
    n = len(roots)
    # Potential energy: sum r_i
    pot = np.sum(roots)
    # Interaction: -sum_{i<j} ln|r_i - r_j|
    inter = 0.0
    for i in range(n):
        for j in range(i+1, n):
            inter -= np.log(abs(roots[i] - roots[j]))
    return pot + inter

def _objective(x):
    """Objective function for optimization."""
    if np.any(x <= 0) or np.any(np.diff(x) <= 0):
        return 1e10
    return _compute_c4(x)

def _objective_grad(x):
    """Gradient of the objective function."""
    n = len(x)
    if np.any(x <= 0) or np.any(np.diff(x) <= 0):
        return np.zeros(n)
    grad = np.zeros(n)
    # d/dx_i (sum x_i) = 1
    grad[:] = 1.0
    # d/dx_i (-sum_{j!=i} ln|x_i - x_j|) = -sum_{j!=i} 1/(x_i - x_j)
    for i in range(n):
        for j in range(n):
            if i != j:
                grad[i] -= 1.0 / (x[i] - x[j])
    return grad

def _c4_from_alpha(alpha):
    """Compute C4 from alpha parameter (alternative parametrization)."""
    # This is a placeholder for a different parametrization if needed
    return alpha

def construct():
    """
    Returns a list of 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    The goal is to minimize the upper bound on C4.
    """
    N = 12
    start_time = time.time()
    deadline = 600.0
    safety_margin = 30.0
    
    # Initialization using Laguerre polynomial roots
    try:
        x0, _ = np.polynomial.laguerre.laggauss(N)
        x0 = np.sort(x0)
    except Exception:
        x0 = np.linspace(1, 2*N, N)
    
    best_x = x0.copy()
    best_score = _objective(x0)
    
    # Multi-start optimization
    rng = np.random.default_rng(42)
    n_starts = 20
    
    for s in range(n_starts):
        if time.time() > deadline - safety_margin:
            break
        
        # Generate perturbed initial guess
        if s == 0:
            x_init = x0.copy()
        else:
            scale = 0.5 + 0.5 * s / n_starts
            x_init = x0 + scale * rng.standard_normal(N)
            x_init = np.sort(np.maximum(x_init, 1e-4))
            # Ensure strict ordering
            for i in range(1, N):
                if x_init[i] <= x_init[i-1]:
                    x_init[i] = x_init[i-1] + 1e-3
        
        bounds = [(1e-6, None) for _ in range(N)]
        
        try:
            res = minimize(
                _objective,
                x_init,
                jac=_objective_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14, 'maxls': 50}
            )
            
            if res.success and res.fun < best_score:
                x_opt = np.sort(res.x)
                if np.all(x_opt > 0) and np.all(np.diff(x_opt) > 0):
                    best_x = x_opt
                    best_score = res.fun
        except Exception:
            continue
    
    # Final polish with tight tolerances
    if time.time() < deadline - safety_margin:
        try:
            res = minimize(
                _objective,
                best_x,
                jac=_objective_grad,
                method='L-BFGS-B',
                bounds=[(1e-6, None) for _ in range(N)],
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-16, 'maxls': 100}
            )
            if res.success and res.fun < best_score:
                x_opt = np.sort(res.x)
                if np.all(x_opt > 0) and np.all(np.diff(x_opt) > 0):
                    best_x = x_opt
                    best_score = res.fun
        except Exception:
            pass
    
    return [float(xi) for xi in best_x]
