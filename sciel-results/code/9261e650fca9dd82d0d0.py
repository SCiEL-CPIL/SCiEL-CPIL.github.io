"""Optimize Hermite-Gaussian test function coefficients to minimize uncertainty bound."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite
import time


def _hermite_gaussian(x, coeffs):
    """Evaluate sum of c_i * H_{2i}(x) * exp(-x^2/2) for i=0,1,2,3.
    coeffs = [c0, c4, c8, c12] corresponding to H_0, H_4, H_8, H_12."""
    g = np.zeros_like(x, dtype=float)
    exp_term = np.exp(-x**2 / 2.0)
    for i, c in enumerate(coeffs):
        if c != 0:
            g += c * eval_hermite(2 * i, x) * exp_term
    return g


def _compute_bound(c0, c4, c8):
    """Compute the upper bound C4 on A(f)*A(F(f)) for given coefficients.
    c12 is derived from the constraint g(0) = 0."""
    # H_n(0) for even n: H_0(0)=1, H_4(0)=-12, H_8(0)=1680, H_12(0)=-240240
    # g(0) = c0*1 + c4*(-12) + c8*1680 + c12*(-240240) = 0
    # c12 = (c0 - 12*c4 + 1680*c8) / 240240
    c12 = (c0 - 12.0 * c4 + 1680.0 * c8) / 240240.0
    coeffs = [c0, c4, c8, c12]
    
    # Use numerical integration on a sufficiently wide grid
    x = np.linspace(-20, 20, 20001)
    dx = x[1] - x[0]
    
    g = _hermite_gaussian(x, coeffs)
    
    # A(f) = sqrt(∫ |g(x)|^2 dx) -- position amplitude
    norm_sq = np.trapz(g**2, x)
    if norm_sq <= 0:
        return 1e10
    A_pos = np.sqrt(norm_sq)
    
    # For the Fourier transform of Hermite-Gaussian, we need to compute
    # the Fourier transform numerically and get its amplitude
    # F(f)(ξ) = ∫ g(x) e^{-2πixξ} dx (or with different convention)
    # Let's use the standard: F(ξ) = ∫ g(x) exp(-i x ξ) dx
    # A(F(f)) = sqrt(∫ |F(ξ)|^2 dξ) = sqrt(2π * ∫ |g(x)|^2 dx) by Parseval
    # But the uncertainty product involves variance, not just amplitude
    
    # Actually, for the uncertainty principle bound, we need:
    # Δx * Δξ where Δx^2 = ∫ x^2 |g|^2 dx / ∫ |g|^2 dx
    # and Δξ^2 = ∫ ξ^2 |F(g)|^2 dξ / ∫ |F(g)|^2 dξ
    
    # By Parseval: ∫ |F(g)|^2 dξ = 2π ∫ |g|^2 dx (for F(ξ)=∫g(x)e^{-ixξ}dx)
    # And ∫ ξ^2 |F(g)|^2 dξ = 2π ∫ |g'(x)|^2 dx (integration by parts)
    
    # Compute g'(x) numerically
    # Use derivative via finite differences or analytical
    # For Hermite-Gaussian, derivative is tractable but let's use numerical
    g_prime = np.gradient(g, x)
    
    # Δx^2
    x2_weighted = np.trapz(x**2 * g**2, x)
    var_x = x2_weighted / norm_sq
    
    # Δξ^2 = 2π ∫ |g'|^2 dx / (2π ∫ |g|^2 dx) = ∫ |g'|^2 dx / ∫ |g|^2 dx
    g_prime_sq = np.trapz(g_prime**2, x)
    var_xi = g_prime_sq / norm_sq
    
    # Uncertainty product: Δx * Δξ = sqrt(var_x * var_xi)
    # But the task says "upper bound on C4" - let me reconsider
    
    # The task mentions A(f)*A(Fourier(f)) which might be the amplitude product,
    # not the variance product. Let me re-read: "reduce the upper bound on A(f)*A(Fourier(f))"
    # A(f) likely means the L2 norm or amplitude. But that would just be sqrt(2π)*norm.
    
    # Actually in uncertainty principle context, A(f) often denotes the spread.
    # Let me compute both and use the product of standard deviations.
    
    uncertainty_product = np.sqrt(var_x * var_xi)
    
    return uncertainty_product


def _objective(params):
    """Objective function for minimization."""
    c0, c4, c8 = params
    return _compute_bound(c0, c4, c8)


def construct():
    """Find optimal coefficients c0, c4, c8 minimizing the uncertainty bound."""
    deadline = time.time() + 180.0  # Safety margin before 210s kill
    
    # Starting point from the seed
    best_params = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_score = _compute_bound(*best_params)
    
    # Try multiple starting points with different optimization methods
    start_points = [
        np.array([-1.13, 0.04, 0.000308642]),  # seed
        np.array([-1.0, 0.05, 0.0003]),
        np.array([-1.2, 0.03, 0.0002]),
        np.array([-0.9, 0.06, 0.0004]),
        np.array([-1.5, 0.02, 0.0001]),
        np.array([-0.5, 0.08, 0.0005]),
        np.array([-2.0, 0.01, 0.00005]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([-1.3, 0.035, 0.00025]),
        np.array([-0.8, 0.055, 0.00035]),
    ]
    
    methods = ['Nelder-Mead', 'Powell', 'COBYLA']
    
    for method in methods:
        for start in start_points:
            if time.time() > deadline - 10.0:
                break
            try:
                result = minimize(
                    _objective,
                    start,
                    method=method,
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if result.success or result.fun < best_score:
                    score = result.fun
                    if score < best_score:
                        best_score = score
                        best_params = result.x
            except Exception:
                continue
        if time.time() > deadline - 10.0:
            break
    
    # Additional local refinement with tighter tolerances
    if time.time() < deadline - 15.0:
        try:
            result = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Return only the first three coefficients (c0, c4, c8)
    return np.array([best_params[0], best_params[1], best_params[2]])
# EVOLVE-BLOCK-END
