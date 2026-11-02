"""Optimized program for the uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution


def _compute_upper_bound(coeffs):
    """
    Compute the upper bound on C4 for a given set of Hermite coefficients.
    
    The test function is g(x) = (c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)) * exp(-x^2/2)
    where c12 is chosen so that g(0) = 0.
    
    The upper bound on A(f)*A(F) is related to the ratio:
        C4_bound = ||g||_4^4 / (||g||_2^2 * ||F(g)||_2^2)
    
    For Gaussian-weighted Hermite polynomials, we can compute these norms analytically
    or numerically.
    
    The key insight: for the uncertainty inequality, we want to minimize the upper bound.
    The upper bound C4 is given by:
        C4 = (∫|g|^4 dx) / (∫|g|^2 dx * ∫|F(g)|^2 dx)
    
    By Plancherel, ∫|F(g)|^2 dx = ∫|g|^2 dx (with appropriate normalization).
    So C4 = (∫|g|^4 dx) / (∫|g|^2 dx)^2
    
    We need to compute these integrals for g(x) = sum(c_i * H_i(x)) * exp(-x^2/2).
    """
    c0, c4, c8 = coeffs
    
    # Compute c12 such that g(0) = 0
    # H_n(0) values: H0(0)=1, H4(0)=-3, H8(0)=105, H12(0)=-10395
    # g(0) = c0*1 + c4*(-3) + c8*105 + c12*(-10395) = 0
    # c12 = (c0 - 3*c4 + 105*c8) / 10395
    c12 = (c0 - 3 * c4 + 105 * c8) / 10395.0
    
    # Now compute the norms numerically
    # Use Gauss-Hermite quadrature or direct integration
    
    # For efficiency, use the fact that we're integrating polynomials times exp(-x^2)
    # g(x) = P(x) * exp(-x^2/2) where P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    # |g|^2 = P(x)^2 * exp(-x^2)
    # |g|^4 = P(x)^4 * exp(-2x^2)
    
    # Let's use numerical integration with a fine grid
    x = np.linspace(-10, 10, 10000)
    dx = x[1] - x[0]
    
    # Hermite polynomials (physicists' convention, but let's use the standard ones)
    # H0(x) = 1
    # H4(x) = x^4 - 6x^2 + 3
    # H8(x) = x^8 - 28x^6 + 420x^4 - 1680x^2 + 105
    # H12(x) = x^12 - 84x^10 + 2520x^8 - 25200x^6 + 63504x^4 - 30240x^2 + 10395
    
    H0 = np.ones_like(x)
    H4 = x**4 - 6*x**2 + 3
    H8 = x**8 - 28*x**6 + 420*x**4 - 1680*x**2 + 105
    H12 = (x**12 - 84*x**10 + 2520*x**8 - 25200*x**6 + 
           63504*x**4 - 30240*x**2 + 10395)
    
    P = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    
    # |g|^2 = P^2 * exp(-x^2)
    g2 = P**2 * np.exp(-x**2)
    # |g|^4 = P^4 * exp(-2x^2)
    g4 = P**4 * np.exp(-2*x**2)
    
    # Integrate
    norm2_sq = np.trapz(g2, x)
    norm4 = np.trapz(g4, x)
    
    if norm2_sq <= 0:
        return np.inf
    
    C4_bound = norm4 / (norm2_sq ** 2)
    
    return C4_bound


def construct():
    """
    Find optimal coefficients for H0, H4, H8 that minimize the upper bound on C4.
    The H12 coefficient is fixed to ensure g(0) = 0.
    """
    # Initial guess from prior state of the art
    x0 = np.array([-113/100, 1/25, 1/3240])
    
    # Define objective function
    def objective(coeffs):
        return _compute_upper_bound(coeffs)
    
    # Use differential evolution for global optimization
    # Bounds for each coefficient
    bounds = [(-5, 5), (-1, 1), (-0.1, 0.1)]
    
    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=42,
        maxiter=100,
        tol=1e-8,
        polish=True
    )
    
    if result.success:
        optimal_coeffs = result.x
    else:
        # Fall back to initial guess if optimization fails
        optimal_coeffs = x0
    
    return optimal_coeffs
# EVOLVE-BLOCK-END
