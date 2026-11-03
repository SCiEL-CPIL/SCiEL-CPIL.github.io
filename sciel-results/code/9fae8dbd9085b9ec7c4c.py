"""Optimize Hermite-Gaussian coefficients to minimize A(g)*A(Fourier(g)).

Uses analytic Fourier transform of Hermite-Gaussian functions:
FT[H_n(x) exp(-x^2/2)] = (-i)^n H_n(xi) exp(-xi^2/2) (up to 2pi convention).

This avoids slow numerical FT and enables fast, accurate objective evaluation.
"""
# EVOLVE-BLOCK-START
import numpy as np


def _phys_hermite(n, x):
    """Physicists' Hermite polynomial H_n(x) via recurrence."""
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x
    h0 = np.ones_like(x)
    h1 = 2.0 * x
    for k in range(2, n + 1):
        h2 = 2.0 * x * h1 - 2.0 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1


def _build_g(coeffs, x):
    """Return g(x) = (a0 H0 + a4 H4 + a8 H8 + a12 H12) * exp(-x^2/2).

    a12 is chosen so that g(0) = 0.
    """
    a0, a4, a8 = coeffs
    H0 = _phys_hermite(0, x)
    H4 = _phys_hermite(4, x)
    H8 = _phys_hermite(8, x)
    H12 = _phys_hermite(12, x)
    H0_0, H4_0, H8_0, H12_0 = 1.0, 3.0, 105.0, 10395.0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    poly = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
    return poly * np.exp(-0.5 * x * x)


def _build_g_hat(coeffs, xi):
    """Return Fourier transform of g using analytic Hermite-Gaussian FT.

    For the convention g_hat(xi) = int g(x) exp(-2*pi*i*x*xi) dx,
    the FT of H_n(x) exp(-x^2/2) involves H_n with scaling.
    Using the unitary convention: FT[H_n(x) e^{-x^2/2}] = (-i)^n H_n(xi) e^{-xi^2/2}
    For the 2pi convention, we need to account for scaling.
    
    Actually, let's use: if f(x) = H_n(x) e^{-x^2/2}, then
    f_hat(xi) = sqrt(2*pi) * (-i)^n * H_n(sqrt(2*pi)*xi) * e^{-2*pi^2*xi^2}
    
    Let me just compute it numerically but efficiently using the fact that
    the FT of a Hermite-Gaussian is another Hermite-Gaussian.
    
    For simplicity and accuracy, I'll use the known result:
    Under the convention F[f](xi) = integral f(x) e^{-2*pi*i*x*xi} dx,
    F[H_n(x) e^{-x^2/2}](xi) = (2*pi)^{1/2} * (-i)^n * H_n(2*pi*xi) * e^{-2*pi^2*xi^2}
    
    Wait, let me be more careful. The standard result is:
    F[e^{-x^2/2}](xi) = sqrt(2*pi) * e^{-2*pi^2*xi^2}
    And for Hermite: F[H_n(x) e^{-x^2/2}](xi) = (2*pi)^{1/2} * (2*pi*i)^n * H_n(2*pi*xi) * e^{-2*pi^2*xi^2}
    
    Actually the cleanest approach: use the unitary FT convention where
    F[f](xi) = (1/sqrt(2*pi)) integral f(x) e^{-i*x*xi} dx
    Then F[H_n(x) e^{-x^2/2}](xi) = (-i)^n H_n(xi) e^{-xi^2/2}
    
    Since we're computing A(g)*A(g_hat) and the constant factors from the
    FT convention will appear in both or cancel in a consistent way, let's
    use the unitary convention for simplicity.
    """
    a0, a4, a8 = coeffs
    H0_0, H4_0, H8_0, H12_0 = 1.0, 3.0, 105.0, 10395.0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    # Under unitary FT: F[H_n(x) e^{-x^2/2}] = (-i)^n H_n(xi) e^{-xi^2/2}
    H0 = _phys_hermite(0, xi)
    H4 = _phys_hermite(4, xi)
    H8 = _phys_hermite(8, xi)
    H12 = _phys_hermite(12, xi)
    gauss = np.exp(-0.5 * xi * xi)
    
    # (-i)^0 = 1, (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = (-i)^4 * (-i)^8 = 1*1 = 1
    # All even powers of (-i) that are multiples of 4 give 1.
    # (-i)^0 = 1, (-i)^4 = ((-i)^2)^2 = (-1)^2 = 1, (-i)^8 = 1, (-i)^12 = 1
    g_hat = (a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12) * gauss
    return g_hat


def _objective(coeffs):
    """Uncertainty concentration: A(g) * A(Fourier(g)) using analytic FT."""
    x = np.linspace(-10.0, 10.0, 8192)
    g = _build_g(coeffs, x)
    A_g = np.trapz(np.abs(g), x)
    
    xi = np.linspace(-10.0, 10.0, 8192)
    g_hat = _build_g_hat(coeffs, xi)
    A_ghat = np.trapz(np.abs(g_hat), xi)
    
    return A_g * A_ghat


def construct():
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    try:
        from scipy.optimize import minimize
        base = _objective(seed)
        best = seed
        best_val = base
        
        rng = np.random.default_rng(0)
        # More aggressive multi-start with varied scales
        scales = [0.01, 0.05, 0.1, 0.2, 0.5]
        for attempt in range(12):
            if attempt == 0:
                x0 = seed
            else:
                scale = scales[(attempt - 1) % len(scales)]
                x0 = seed + rng.normal(0.0, scale, size=3)
            
            res = minimize(
                _objective,
                x0,
                method="L-BFGS-B",
                options={"maxiter": 300, "ftol": 1e-14, "gtol": 1e-10},
            )
            if res.fun < best_val and np.all(np.isfinite(res.x)):
                cand = np.asarray(res.x, dtype=float)
                v = _objective(cand)
                if np.isfinite(v) and v < best_val:
                    best_val = v
                    best = cand
        
        return best
    except Exception:
        return seed
# EVOLVE-BLOCK-END
