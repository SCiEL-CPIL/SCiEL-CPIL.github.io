"""Optimized program for the uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial
import warnings


def _compute_bound_analytic(c0, c4, c8):
    """
    Compute the uncertainty product analytically using Hermite function expansion.
    
    g(x) = (c0 H0 + c4 H4 + c8 H8 + c12 H12) exp(-x^2/2)
    Let phi_n(x) be orthonormal Hermite functions.
    H_n(x) exp(-x^2/2) = A_n phi_n(x) where A_n = sqrt(2^n n! sqrt(pi)).
    
    g(x) = sum a_n phi_n(x) with a_n = c_n A_n.
    
    Norm squared N = sum a_n^2.
    M_g = int x^2 g^2 dx.
    M_F = int x^2 |F[g]|^2 dx.
    
    Since x^2 connects n to n±2 only, and our indices {0,4,8,12} have no pairs differing by 2,
    M_g = sum_n a_n^2 (n + 0.5) and M_F = sum_n a_n^2 (n + 0.5).
    
    Objective = (M_g * M_F) / (N * N) = (M_g / N)^2
    """
    # Constants: H_n(0) values
    h0_0 = 1.0
    h4_0 = 12.0
    h8_0 = 1680.0
    h12_0 = 665280.0
    
    c12 = -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0
    
    # Coefficients a_n = c_n * A_n
    # A_n = sqrt(2^n * n! * sqrt(pi))
    A0 = np.sqrt(1.0 * 1.0 * np.sqrt(np.pi))
    A4 = np.sqrt(16.0 * 24.0 * np.sqrt(np.pi))
    A8 = np.sqrt(256.0 * 40320.0 * np.sqrt(np.pi))
    A12 = np.sqrt(4096.0 * float(factorial(12)) * np.sqrt(np.pi))
    
    a0 = c0 * A0
    a4 = c4 * A4
    a8 = c8 * A8
    a12 = c12 * A12
    
    N = a0**2 + a4**2 + a8**2 + a12**2
    if N < 1e-15:
        return 1e10
        
    # D_n = n + 0.5
    D0 = 0.5
    D4 = 4.5
    D8 = 8.5
    D12 = 12.5
    
    M_g = D0 * a0**2 + D4 * a4**2 + D8 * a8**2 + D12 * a12**2
    J = (M_g / N) ** 2
    
    return J


def _objective(coeffs):
    """Objective function for optimization."""
    c0, c4, c8 = coeffs
    try:
        bound = _compute_bound_analytic(c0, c4, c8)
        if not np.isfinite(bound) or bound < 0:
            return 1e10
        return bound
    except Exception:
        return 1e10


def construct():
    """
    Find coefficients that minimize the upper bound on A(f)*A(Fourier(f)).
    
    Returns an array of three coefficients [c0, c4, c8] for the Hermite combination.
    The degree-12 coefficient is determined to ensure g(0) = 0.
    """
    # Start from the prior state of the art and refine
    initial = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Use multiple starting points with L-BFGS-B for local refinement
    best_x = initial
    best_fun = _objective(initial)
    
    # Try several random starting points
    rng = np.random.RandomState(42)
    for i in range(50):
        # Sample from a reasonable range
        x0 = rng.uniform(-10, 10, size=3)
        try:
            res = minimize(_objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
            if res.fun < best_fun:
                best_fun = res.fun
                best_x = res.x
        except Exception:
            pass
    
    # Also try from the original initial point
    try:
        res = minimize(_objective, initial, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
        if res.fun < best_fun:
            best_fun = res.fun
            best_x = res.x
    except Exception:
        pass
    
    return best_x
# EVOLVE-BLOCK-END
