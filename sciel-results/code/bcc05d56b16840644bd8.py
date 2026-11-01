"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

# Deadline guard: 210s wall-clock limit, keep safety margin
START_TIME = time.time()
TIME_LIMIT = 190.0  # seconds, leaving 20s buffer before 210s kill

def _elapsed():
    return time.time() - START_TIME

def _check_deadline():
    return _elapsed() > TIME_LIMIT

def _compute_objective(coeffs):
    """
    Compute the upper bound on A(f)*A(Fourier(f)) for a given set of coefficients.
    
    The test function is g(x) = (c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)) * exp(-x^2/2)
    where c12 is determined by the constraint g(0) = 0.
    
    The uncertainty product bound involves integrals of |g(x)| and |Fourier(g)(xi)|.
    For Hermite-Gaussian functions, we can compute these analytically or via quadrature.
    
    The key insight: for a function g(x) = sum_k c_k H_k(x) e^{-x^2/2}, 
    the L1 norm and the L1 norm of its Fourier transform can be computed.
    
    The uncertainty principle bound A(f)*A(F) >= 1/sqrt(4*pi) for normalized functions,
    but here we're looking at the upper bound that can be minimized by choosing coefficients.
    
    Let's use the fact that for Hermite polynomials:
    - H_0(x) = 1
    - H_4(x) = x^4 - 6x^2 + 3
    - H_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    - H_12(x) = x^12 - 42x^10 + 630x^8 - 7560x^6 + 45360x^4 - 120960x^2 + 10395
    
    The constraint g(0) = 0 gives:
    c0*1 + c4*3 + c8*105 + c12*10395 = 0
    So c12 = -(c0 + 3*c4 + 105*c8) / 10395
    
    The objective is the upper bound on the uncertainty product.
    For a function g, A(g) = (∫|g|^2)^(1/2) * (∫x^2|g|^2)^(1/2) / ∫|g|^2
    Wait, actually the uncertainty product is typically Δx * Δp where:
    Δx^2 = ∫x^2|g|^2 / ∫|g|^2
    Δp^2 = ∫ξ^2|G(ξ)|^2 / ∫|G(ξ)|^2
    
    And by Parseval, ∫|G|^2 = ∫|g|^2 (up to normalization).
    
    For the Hermite-Gaussian basis, the variance can be computed analytically.
    
    Actually, let me think more carefully. The task says "upper bound on A(f)*A(Fourier(f))".
    This is the Heisenberg uncertainty product. For a function g(x) = sum c_k H_k(x) e^{-x^2/2}:
    
    The L2 norm squared: ||g||_2^2 = sum_k c_k^2 * k! * sqrt(pi)
    (since ∫ H_m(x) H_n(x) e^{-x^2} dx = n! sqrt(pi) delta_mn)
    
    The second moment: ∫ x^2 |g|^2 dx
    Using x^2 H_n(x) = H_{n+2}(x)/4 + n H_n(x)/2 ... actually let me use the recurrence.
    
    Actually, for the variance calculation with Hermite-Gaussians, it's easier to just 
    compute the integrals numerically using Gauss-Hermite quadrature or direct integration.
    
    Let me compute:
    - norm_sq = ∫ |g(x)|^2 dx
    - x2_moment = ∫ x^2 |g(x)|^2 dx
    - For the Fourier side, since Fourier of H_n(x) e^{-x^2/2} is related to H_n(xi) e^{-xi^2/2}
      (up to phase and normalization), the xi^2 moment will be similar.
    
    Actually, for the standard Fourier transform convention, if g(x) = H_n(x) e^{-x^2/2},
    then G(xi) = (-i)^n sqrt(2*pi) H_n(xi) e^{-xi^2/2} (with appropriate normalization).
    
    So the structure is symmetric! The xi^2 moment of |G|^2 equals the x^2 moment of |g|^2
    up to the normalization factor.
    
    Let me just compute everything numerically for accuracy.
    """
    c0, c4, c8 = coeffs
    c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
    
    # Hermite polynomials (physicists' convention)
    def H0(x): return np.ones_like(x)
    def H4(x): return x**4 - 6*x**2 + 3
    def H8(x): return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    def H12(x): return x**12 - 42*x**10 + 630*x**8 - 7560*x**6 + 45360*x**4 - 120960*x**2 + 10395
    
    # Evaluate g(x)
    x = np.linspace(-10, 10, 10000)
    dx = x[1] - x[0]
    
    g = (c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)) * np.exp(-x**2/2)
    
    # L2 norm squared
    norm_sq = np.trapz(g**2, x)
    if norm_sq < 1e-15:
        return 1e10  # penalty for zero norm
    
    # x^2 moment
    x2_moment = np.trapz(x**2 * g**2, x)
    
    # Variance in x
    var_x = x2_moment / norm_sq
    
    # For the Fourier transform, use the property that for Hermite-Gaussians,
    # the Fourier transform preserves the structure. 
    # G(xi) = (-i)^n sqrt(2*pi) * (normalized) H_n(xi) e^{-xi^2/2}
    # More precisely, with the convention F[g](xi) = ∫ g(x) e^{-i x xi} dx:
    # F[H_n(x) e^{-x^2/2}](xi) = (-i)^n sqrt(2*pi) H_n(xi) e^{-xi^2/2}
    
    # So |G(xi)|^2 = 2*pi * (sum c_k H_k(xi))^2 e^{-xi^2}
    # (the phases cancel in the magnitude squared)
    
    xi = x  # symmetric grid
    G_sq = 2*np.pi * (c0*H0(xi) + c4*H4(xi) + c8*H8(xi) + c12*H12(xi))**2 * np.exp(-xi**2)
    
    norm_sq_G = np.trapz(G_sq, xi)
    if norm_sq_G < 1e-15:
        return 1e10
    
    xi2_moment = np.trapz(xi**2 * G_sq, xi)
    var_xi = xi2_moment / norm_sq_G
    
    # Uncertainty product: sqrt(var_x) * sqrt(var_xi)
    # But wait, the "upper bound" might refer to something specific.
    # The Heisenberg uncertainty says sqrt(var_x) * sqrt(var_xi) >= 1/2
    # The "upper bound on A(f)*A(F)" that we're minimizing is just this product.
    
    uncertainty_product = np.sqrt(var_x) * np.sqrt(var_xi)
    
    return uncertainty_product

def construct():
    """
    Construct the three coefficients of the Hermite combination (degrees 0, 4, 8).
    The degree-12 weight is fixed so g(0) = 0.
    Returns an array of 3 coefficients.
    """
    # Prior state of the art
    initial_guess = np.array([-113.0/100, 1.0/25, 1.0/3240])
    
    best_coeffs = initial_guess.copy()
    best_val = _compute_objective(best_coeffs)
    
    if _check_deadline():
        return best_coeffs
    
    # Try optimization
    try:
        # Use trust-constr algorithm
        result = minimize(
            _compute_objective,
            initial_guess,
            method='trust-constr',
            options={'maxiter': 200, 'gtol': 1e-8, 'xtol': 1e-10, 'verbose': 0}
        )
        
        if result.success or _check_deadline():
            candidate = result.x
            candidate_val = _compute_objective(candidate)
            if candidate_val < best_val:
                best_coeffs = candidate
                best_val = candidate_val
        
        if not _check_deadline():
            # Try a few different initial points
            for start in [
                np.array([0.0, 0.0, 0.0]),
                np.array([-1.0, 0.1, 0.001]),
                np.array([-0.5, 0.05, 0.0005]),
                np.array([-113.0/100, 1.0/25, 1.0/3240]) + np.array([0.1, -0.01, 0.0001]),
                np.array([-113.0/100, 1.0/25, 1.0/3240]) + np.array([-0.1, 0.01, -0.0001]),
            ]:
                if _check_deadline():
                    break
                
                res = minimize(
                    _compute_objective,
                    start,
                    method='trust-constr',
                    options={'maxiter': 100, 'gtol': 1e-8, 'xtol': 1e-10, 'verbose': 0}
                )
                
                if res.success:
                    cand_val = _compute_objective(res.x)
                    if cand_val < best_val:
                        best_coeffs = res.x.copy()
                        best_val = cand_val
                        
    except Exception:
        pass
    
    if _check_deadline():
        return best_coeffs
    
    # Final polish with tighter tolerance
    try:
        result = minimize(
            _compute_objective,
            best_coeffs,
            method='trust-constr',
            options={'maxiter': 500, 'gtol': 1e-12, 'xtol': 1e-14, 'verbose': 0}
        )
        
        if result.success:
            cand_val = _compute_objective(result.x)
            if cand_val < best_val:
                best_coeffs = result.x.copy()
                best_val = cand_val
                
    except Exception:
        pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
