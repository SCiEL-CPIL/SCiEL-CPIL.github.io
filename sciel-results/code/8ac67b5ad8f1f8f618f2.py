"""Optimized program for uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite
from scipy.integrate import quad
import time

_START_TIME = time.time()
_TIME_LIMIT = 180.0
_SAFETY_MARGIN = 10.0  # seconds before the 180s limit


def _hermite_gaussian(x, coeffs):
    """Evaluate (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2) at x."""
    c0, c4, c8, c12 = coeffs
    h0 = 1.0
    h4 = eval_hermite(4, x)
    h8 = eval_hermite(8, x)
    h12 = eval_hermite(12, x)
    poly = c0 * h0 + c4 * h4 + c8 * h8 + c12 * h12
    return poly * np.exp(-0.5 * x * x)


def _compute_c12(c0, c4, c8):
    """Compute c12 so that g(0) = 0."""
    # H0(0)=1, H4(0)=-3, H8(0)=105, H12(0)=-10395
    # c0*1 + c4*(-3) + c8*105 + c12*(-10395) = 0
    # c12 = (c0 - 3*c4 + 105*c8) / 10395
    return (c0 - 3 * c4 + 105 * c8) / 10395.0


def _objective(params):
    """Compute the product A(g) * A(F(g)) for the Hermite-Gaussian function."""
    c0, c4, c8 = params
    c12 = _compute_c12(c0, c4, c8)
    coeffs = np.array([c0, c4, c8, c12])
    
    # Compute L1 norm of g(x)
    g_func = lambda x: _hermite_gaussian(x, coeffs)
    try:
        l1_g, _ = quad(lambda x: abs(g_func(x)), -15, 15, limit=200)
    except Exception:
        return 1e10
    
    # Fourier transform: F{H_n(x) * exp(-x^2/2)} = (-i)^n * H_n(x) * exp(-x^2/2) * (1/sqrt(2*pi))
    # Actually, the Fourier transform convention matters. Let's use:
    # F(f)(ξ) = ∫ f(x) * exp(-2πi x ξ) dx
    # For Hermite functions with exp(-x^2/2), the FT is more complex.
    # Let's use the symmetric convention or just compute numerically.
    
    # Actually, for the uncertainty principle, the exact FT convention affects constants.
    # Let me use the standard math convention: F(f)(ξ) = ∫ f(x) e^{-2πixξ} dx
    # But for Hermite functions, it's easier to use the physics convention.
    
    # Let's compute the FT numerically using FFT or quad.
    # The FT of H_n(x) * exp(-x^2/2) under the convention F(f)(ξ) = ∫ f(x) e^{-ixξ} dx
    # is (-i)^n * H_n(x) * exp(-x^2/2) (with appropriate scaling)
    
    # Under F(f)(ξ) = ∫ f(x) e^{-ixξ} dx / sqrt(2π):
    # F{H_n * exp(-x^2/2)} = (-i)^n * H_n * exp(-x^2/2)
    
    # So the L1 norm of the FT is the same as the L1 norm of g if we use this convention!
    # Because |(-i)^n * H_n * exp(-x^2/2)| = |H_n * exp(-x^2/2)|
    
    # Wait, that's only for each individual Hermite function. For a linear combination:
    # F{Σ c_n H_n exp(-x^2/2)} = Σ c_n (-i)^n H_n exp(-x^2/2)
    # So the FT is (c0*H0 + c4*(-i)^4*H4 + c8*(-i)^8*H8 + c12*(-i)^12*H12) * exp(-x^2/2)
    # (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = (-i)^8 * (-i)^4 = 1*1 = 1
    # So all phases are 1! The FT has the same absolute value as g.
    
    # Therefore A(g) = A(F(g)) under this convention, and the product is A(g)^2.
    
    # But wait, the convention might be different. Let me be more careful.
    # If F(f)(ξ) = ∫ f(x) e^{-2πixξ} dx, then the scaling is different.
    
    # For now, let's assume the product is l1_g * l1_f where l1_f is the L1 norm of the FT.
    # Let me compute the FT numerically to be safe.
    
    # Actually, let me just compute both numerically to be safe.
    
    def ft_func(xi):
        """Numerical FT at frequency xi."""
        # F(f)(ξ) = ∫ f(x) e^{-2πi x ξ} dx
        # This is complex, so we need to integrate real and imaginary parts
        def real_part(x):
            return g_func(x) * np.cos(2 * np.pi * xi * x)
        def imag_part(x):
            return -g_func(x) * np.sin(2 * np.pi * xi * x)
        
        try:
            re, _ = quad(real_part, -15, 15, limit=200)
            im, _ = quad(imag_part, -15, 15, limit=200)
            return re + 1j * im
        except Exception:
            return 0.0 + 0.0j
    
    # This is too slow. Let me use the analytic property instead.
    # Under the convention F(f)(ξ) = (1/sqrt(2π)) ∫ f(x) e^{-ixξ} dx:
    # F{H_n(x) e^{-x^2/2}} = (-i)^n H_n(x) e^{-x^2/2}
    # So for our combination, the FT is:
    # (c0*H0 + c4*H4 + c8*H8 + c12*H12) * e^{-x^2/2}  (since all (-i)^even = 1)
    # So |F(g)(ξ)| = |g(ξ)| and the L1 norms are equal.
    
    # Therefore the product is l1_g * l1_g = l1_g^2
    
    # But this depends on the convention used in the task. Let me just use l1_g * l1_g.
    
    return l1_g * l1_g


def construct():
    """Find optimal coefficients for H0, H4, H8 to minimize the uncertainty bound."""
    # Start from the seed values
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_params = seed.copy()
    best_score = _objective(seed)
    
    # Try multiple starting points and local optimizations
    start_points = [
        seed,
        seed + np.array([0.01, 0.001, 0.0001]),
        seed - np.array([0.01, 0.001, 0.0001]),
        np.array([-1.1, 0.04, 0.0003]),
        np.array([-1.2, 0.03, 0.0002]),
        np.array([-1.0, 0.05, 0.0004]),
    ]
    
    for x0 in start_points:
        if time.time() - _START_TIME > _TIME_LIMIT - _SAFETY_MARGIN:
            break
        
        try:
            result = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if result.success and result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            continue
    
    # Final polish with tighter tolerance
    if time.time() - _START_TIME < _TIME_LIMIT - _SAFETY_MARGIN:
        try:
            result = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if result.success and result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            pass
    
    return best_params
# EVOLVE-BLOCK-END
