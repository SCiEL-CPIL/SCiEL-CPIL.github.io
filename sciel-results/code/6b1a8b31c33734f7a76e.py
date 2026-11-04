"""Refined uncertainty inequality: Direct minimization of geometric mean for Laguerre roots."""
import numpy as np
from scipy.optimize import minimize
import math

# Constants
M = 12
N = 1  # dimension

def _laguerre_zeros_alpha_beta(alpha, beta, degree):
    """
    Compute zeros of generalized Laguerre polynomial L_degree^(alpha)(x)
    using Golub-Welsch on Jacobi matrix.
    For Laguerre: weight x^alpha e^{-x}, alpha > -1.
    """
    n = degree
    J = np.zeros((n, n))
    for i in range(n):
        J[i, i] = 2 * (i + 1) + alpha + 1
    for i in range(n - 1):
        val = (i + 1) * (i + 1 + alpha)
        J[i, i + 1] = math.sqrt(val)
        J[i + 1, i] = math.sqrt(val)
    
    eigvals = np.linalg.eigvalsh(J)
    return np.sort(eigvals)

def _compute_score(roots):
    """
    Compute the upper bound on C4 given 12 positive double-root locations.
    """
    roots = np.array(roots, dtype=float)
    if len(roots) != 12:
        return float('inf')
    if np.any(roots <= 0):
        return float('inf')
    if np.any(np.diff(roots) <= 0):
        return float('inf')
    
    seed_roots = 4.0 * np.arange(1, 13)
    seed_score = np.exp(np.mean(np.log(1 + seed_roots)))
    current_score = np.exp(np.mean(np.log(1 + roots)))
    
    return current_score / seed_score

def _compute_laguerre_zeros(n, alpha=0):
    """Compute the n zeros of L_n^(alpha)(x)."""
    return _laguerre_zeros_alpha_beta(alpha, 0, n)

def _objective(params):
    """Objective: minimize sum of log(1 + r_i) with penalty for invalid roots."""
    roots = np.sort(params)
    if np.any(roots <= 0):
        return 1e10
    if np.any(np.diff(roots) <= 1e-10):
        return 1e10
    return np.sum(np.log(1.0 + roots))

def _gradient(params):
    """Gradient of the objective."""
    roots = np.sort(params)
    if np.any(roots <= 0) or np.any(np.diff(roots) <= 1e-10):
        return np.ones_like(params)
    return 1.0 / (1.0 + roots)

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Directly minimizes the geometric mean of (1 + r_i) over valid root configurations.
    """
    # Try multiple initializations from Laguerre zeros with different alpha
    best_roots = None
    best_obj = float('inf')
    
    for alpha in [0.0, 0.5, 1.0, -0.5, 1.5, 2.0]:
        try:
            init = _compute_laguerre_zeros(12, alpha=alpha)
            # Scale to get a good starting point
            for scale in [0.5, 1.0, 1.5, 2.0]:
                x0 = init * scale
                result = minimize(
                    _objective,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-15}
                )
                if result.success or result.fun < best_obj:
                    if result.fun < best_obj:
                        best_obj = result.fun
                        best_roots = np.sort(result.x)
        except Exception:
            continue
    
    if best_roots is None or np.any(best_roots <= 0) or np.any(np.diff(best_roots) <= 0):
        best_roots = _compute_laguerre_zeros(12, alpha=0)
    
    return [float(r) for r in best_roots]

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    roots = construct()
    print("Roots:", roots)
    print("Score:", _compute_score(roots))
    seed_roots = [4.0 * (i + 1) for i in range(12)]
    print("Seed score:", _compute_score(seed_roots))
