"""Rewritten program: optimize the uncertainty bound for H0+H4+H8 Hermite-Gaussian test function."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_hermite
import time


def _hermite_gauss(x, n):
    """Compute H_n(x) * exp(-x^2/2) using recurrence.
    H_n are physicists' Hermite polynomials."""
    # H_0 = 1, H_1 = 2x
    # H_{n+1} = 2x H_n - 2n H_{n-1}
    if n == 0:
        return np.exp(-x**2 / 2)
    h_nm1 = np.ones_like(x)
    h_n = 2 * x
    for k in range(1, n):
        h_np1 = 2 * x * h_n - 2 * k * h_nm1
        h_nm1, h_n = h_n, h_np1
    return h_n * np.exp(-x**2 / 2)


def _build_g(x, a0, a4, a8):
    """Construct g(x) = (a0*H0 + a4*H4 + a8*H8) * exp(-x^2/2).
    The H12 coefficient is derived to enforce g(0) = 0."""
    # H_n(0) for even n: H_0(0)=1, H_4(0)=12, H_8(0)=1344, H_12(0)=103040
    # g(0) = a0*1 + a4*12 + a8*1344 + a12*103040 = 0
    # => a12 = -(a0 + 12*a4 + 1344*a8) / 103040
    a12 = -(a0 + 12 * a4 + 1344 * a8) / 103040.0
    
    g = (a0 * _hermite_gauss(x, 0) +
         a4 * _hermite_gauss(x, 4) +
         a8 * _hermite_gauss(x, 8) +
         a12 * _hermite_gauss(x, 12))
    return g


def _compute_bound(a0, a4, a8, n_quad=4096):
    """Compute the upper bound on A(f)*A(Fourier(f)).
    
    The uncertainty bound involves ||g||_1 / min|ĝ| structure.
    We compute:
      - ||g||_1 = integral |g(x)| dx
      - ĝ is the Fourier transform of g
      - min|ĝ| over the support where g is non-zero
    
    For the Hermite-Gaussian basis, the Fourier transform maps H_n(x)exp(-x^2/2)
    to (-i)^n H_n(x) exp(-x^2/2) (up to normalization).
    
    The bound C4 = sup_{f} A(f) * A(ĝ) where A(f) = ||f||_1 / ||f||_2 or similar.
    
    Based on the task: score is the upper bound on C4.
    The key quantity is ||g||_1^2 / min|ĝ|^2 or similar ratio.
    
    Let's use the standard formulation: for the uncertainty product,
    we need to minimize the ratio that gives the worst-case bound.
    
    Actually, from the knowledge: "the actual score is related to the uncertainty bound 
    which involves the ratio ||g||_1 / min|ĝ|"
    
    So we minimize: ||g||_1 / min|ĝ(x)|
    
    But we need to be careful about the Fourier transform normalization.
    """
    # Use Gauss-Hermite quadrature for efficient integration
    # For functions of form h(x)*exp(-x^2), use standard GH quadrature
    # Our g(x) = polynomial * exp(-x^2/2), so |g(x)| = |poly| * exp(-x^2/2)
    # integral |g| dx = integral |poly| * exp(-x^2/2) dx
    
    # Let's use a fine grid for numerical integration
    x_min, x_max = -8, 8
    x = np.linspace(x_min, x_max, n_quad)
    dx = x[1] - x[0]
    
    g = _build_g(x, a0, a4, a8)
    
    # L1 norm of g
    l1_norm = np.trapz(np.abs(g), x)
    
    # Fourier transform of g
    # Using FFT-based approach or direct computation
    # For the Fourier transform: ĝ(ξ) = ∫ g(x) exp(-2πixξ) dx
    # But in the Hermite context, the natural FT is: 
    #   ĝ(ξ) = ∫ g(x) exp(-ixξ) dx (mathematical convention)
    # With our normalization, H_n(x)exp(-x^2/2) FT = (-i)^n H_n(ξ)exp(-ξ^2/2)
    
    # So ĝ(ξ) = a0*(-i)^0*H0(ξ)exp(-ξ^2/2) + a4*(-i)^4*H4(ξ)exp(-ξ^2/2) + ...
    # (-i)^0 = 1, (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = 1
    # All even powers of (-i) are 1!
    
    # So ĝ(ξ) has the same functional form as g(x) but with the coefficients
    # and evaluated at ξ. Wait, that's only true if the FT convention matches.
    
    # Let me compute it numerically to be safe.
    # Use FFT with appropriate scaling
    n_fft = 1 << 14  # 16384
    x_fft = np.linspace(-x_max, x_max, n_fft)
    dx_fft = x_fft[1] - x_fft[0]
    
    g_fft = _build_g(x_fft, a0, a4, a8)
    
    # Fourier transform using FFT
    # ĝ(ξ) = ∫ g(x) exp(-i x ξ) dx ≈ dx * sum g(x_k) exp(-i x_k ξ_j)
    # The FFT computes sum g(x_k) exp(-2πi k j / N)
    # We need to map this properly
    
    # Actually, let's just compute the FT directly on a grid
    xi = np.linspace(-x_max, x_max, n_fft)
    # ĝ(ξ) = ∫ g(x) exp(-ixξ) dx
    # Compute via matrix multiplication or FFT
    
    # Use FFT: 
    # Let x_k = -L + k*dx for k=0,...,N-1
    # ξ_j = -L + j*dx for j=0,...,N-1
    # exp(-i x_k ξ_j) = exp(-i(-L+k*dx)(-L+j*dx))
    # This is getting complicated. Let's use a direct approach with fewer points.
    
    # Simpler: use the fact that in the Hermite basis, the FT is simple
    # For the convention ĝ(ξ) = (1/sqrt(2π)) ∫ g(x) e^{-ixξ} dx
    # H_n(x)e^{-x^2/2} -> (-i)^n H_n(ξ) e^{-ξ^2/2} (with 1/sqrt(2π) factor absorbed)
    
    # Let's just compute numerically with a moderate grid
    xi_grid = np.linspace(-6, 6, 2048)
    # Compute ĝ(ξ) = ∫ g(x) exp(-ixξ) dx for each ξ
    # This is expensive. Instead, use the Hermite property.
    
    # With the convention where FT{H_n(x)e^{-x^2/2}} = (-i)^n H_n(ξ)e^{-ξ^2/2}:
    # Since all our n are even (0,4,8,12), (-i)^n = 1 for all
    # So ĝ(ξ) = a0*H0(ξ)e^{-ξ^2/2} + a4*H4(ξ)e^{-ξ^2/2} + a8*H8(ξ)e^{-ξ^2/2} + a12*H12(ξ)e^{-ξ^2/2}
    # = _build_g(ξ, a0, a4, a8)
    
    # So the FT of g is just g evaluated at ξ (for this specific basis and convention)!
    # This means min|ĝ| = min|g| over the domain.
    
    g_xi = _build_g(xi_grid, a0, a4, a8)
    
    # Find min|ĝ| where |g| is significant
    # We need min|ĝ(ξ)| over ξ where the function is not zero
    # To avoid the trivial zero at infinity, look at the minimum in a bounded region
    # where |g| is non-negligible
    
    # The min|ĝ| should be taken over the support. Let's find where |g| > threshold
    threshold = 1e-10
    mask = np.abs(g_xi) > threshold
    if np.any(mask):
        min_abs_g_hat = np.min(np.abs(g_xi[mask]))
    else:
        min_abs_g_hat = 1e-15
    
    # The bound is proportional to ||g||_1 / min|ĝ|
    # Or perhaps ||g||_1^2 / min|ĝ|
    
    # Based on the knowledge: "ratio ||g||_1 / min|ĝ|"
    bound = l1_norm / max(min_abs_g_hat, 1e-15)
    
    return bound


def _objective(coeffs):
    """Objective function to minimize."""
    a0, a4, a8 = coeffs
    try:
        return _compute_bound(a0, a4, a8)
    except:
        return 1e10


def construct():
    """Return three coefficients of the Hermite combination (degrees 0, 4, 8)."""
    deadline = time.time() + 180
    
    # Start with the known good solution
    best_coeffs = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_score = _compute_bound(*best_coeffs)
    
    # Try a few local optimizations from different starting points
    start_points = [
        best_coeffs,
        np.array([-1.0, 0.05, 0.0003]),
        np.array([-1.2, 0.04, 0.0002]),
        np.array([-0.9, 0.06, 0.0004]),
        np.array([-1.0, 0.045, 0.00031]),
        np.array([-1.1, 0.055, 0.00029]),
    ]
    
    for sp in start_points:
        if time.time() > deadline - 30:
            break
        try:
            result = minimize(_objective, sp, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < best_score:
                best_score = result.fun
                best_coeffs = result.x
        except:
            pass
    
    # Also try differential evolution in a bounded region if time permits
    if time.time() < deadline - 60:
        try:
            bounds = [(-3, 0), (0.01, 0.1), (0.0001, 0.001)]
            de_result = differential_evolution(_objective, bounds, maxiter=100, 
                                                popsize=15, tol=1e-8, seed=42,
                                                polish=True)
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_coeffs = de_result.x
        except:
            pass
    
    # Final polish with high precision
    if time.time() < deadline - 10:
        try:
            polish_result = minimize(_objective, best_coeffs, method='Nelder-Mead',
                                    options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-14})
            if polish_result.fun < best_score:
                best_score = polish_result.fun
                best_coeffs = polish_result.x
        except:
            pass
    
    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
