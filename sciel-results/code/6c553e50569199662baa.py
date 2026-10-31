"""Program for the uncertainty inequality task using Remez-like optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import hermite
import math


def _hermite_poly(n, x):
    """Compute the nth physicist's Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    elif n == 2:
        return 4 * x**2 - 2
    elif n == 3:
        return 8 * x**3 - 12 * x
    elif n == 4:
        return 16 * x**4 - 48 * x**2 + 12
    elif n == 5:
        return 32 * x**5 - 160 * x**3 + 120 * x
    elif n == 6:
        return 64 * x**6 - 480 * x**4 + 720 * x**2 - 120
    elif n == 7:
        return 128 * x**7 - 1344 * x**5 + 3360 * x**3 - 1680 * x
    elif n == 8:
        return 256 * x**8 - 3584 * x**6 + 13440 * x**4 - 13440 * x**2 + 1680
    elif n == 12:
        # H_12(x) = 4096 x^12 - 122880 x^10 + 1290240 x^8 - 6451200 x^6 + 13107200 x^4 - 9031680 x^2 + 1032192
        return (4096 * x**12 - 122880 * x**10 + 1290240 * x**8 
                - 6451200 * x**6 + 13107200 * x**4 - 9031680 * x**2 + 1032192)
    else:
        # Fallback using scipy
        return hermite(n)(x)


def _compute_bound(c0, c4, c8):
    """
    Compute the upper bound C4 on A(f)*A(F) for the given Hermite coefficients.
    
    The test function is g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    where c12 is chosen so that g(0) = 0.
    
    H0(0) = 1, H4(0) = 12, H8(0) = 1680, H12(0) = 1032192
    
    So c12 * 1032192 = -(c0 + 12*c4 + 1680*c8)
    c12 = -(c0 + 12*c4 + 1680*c8) / 1032192
    """
    c12 = -(c0 + 12 * c4 + 1680 * c8) / 1032192.0
    
    # We need to compute the bound on A(f) * A(F)
    # A(f) = sup_x |x f(x)| and A(F) = sup_x |xi F(xi)| where F is the Fourier transform
    # For this specific problem, the bound involves evaluating certain integrals or suprema
    
    # The standard approach: the bound C4 is determined by the worst-case ratio
    # We need to evaluate the supremum of |x g(x)| * |xi G(xi)| where G is FT of g
    
    # Let's compute this numerically
    # First, build g(x)
    x = np.linspace(-15, 15, 10001)
    gauss = np.exp(-x**2 / 2)
    g = (c0 * _hermite_poly(0, x) + c4 * _hermite_poly(4, x) + 
         c8 * _hermite_poly(8, x) + c12 * _hermite_poly(12, x)) * gauss
    
    # A(f) = sup |x g(x)|
    A_f = np.max(np.abs(x * g))
    
    # Now compute Fourier transform G(xi)
    # Using the fact that FT of H_n(x) exp(-x^2/2) relates to H_n(-xi) exp(-xi^2/2)
    # Actually, for physicist's Hermite: FT[H_n(x) e^{-x^2/2}] = (-i)^n H_n(xi) e^{-xi^2/2} * sqrt(2*pi)
    # Wait, let me be more careful. The standard FT convention: F(xi) = integral f(x) e^{-i x xi} dx
    # FT[e^{-x^2/2}] = sqrt(2*pi) * e^{-xi^2/2}
    # FT[H_n(x) e^{-x^2/2}] = (-i)^n sqrt(2*pi) H_n(xi) e^{-xi^2/2}
    
    xi = np.linspace(-15, 15, 10001)
    gauss_xi = np.exp(-xi**2 / 2)
    
    # FT of H0 * gauss = sqrt(2*pi) * gauss_xi
    # FT of H4 * gauss = (-i)^4 sqrt(2*pi) H4(xi) gauss_xi = sqrt(2*pi) H4(xi) gauss_xi
    # FT of H8 * gauss = (-i)^8 sqrt(2*pi) H8(xi) gauss_xi = sqrt(2*pi) H8(xi) gauss_xi
    # FT of H12 * gauss = (-i)^12 sqrt(2*pi) H12(xi) gauss_xi = sqrt(2*pi) H12(xi) gauss_xi
    
    sqrt_2pi = math.sqrt(2 * math.pi)
    G = sqrt_2pi * (c0 * _hermite_poly(0, xi) + c4 * _hermite_poly(4, xi) + 
                    c8 * _hermite_poly(8, xi) + c12 * _hermite_poly(12, xi)) * gauss_xi
    
    # A(F) = sup |xi G(xi)|
    A_F = np.max(np.abs(xi * G))
    
    return A_f * A_F


def construct():
    """
    Return three coefficients [c0, c4, c8] that minimize the upper bound C4.
    """
    # Start from the known good solution
    best_c = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_bound = _compute_bound(*best_c)
    
    # Use Nelder-Mead or similar optimization
    from scipy.optimize import minimize
    
    def objective(c):
        try:
            return _compute_bound(c[0], c[1], c[2])
        except Exception:
            return 1e10
    
    # Try multiple starting points
    start_points = [
        best_c,
        np.array([-1.0, 0.1, 0.001]),
        np.array([-1.2, 0.05, 0.0005]),
        np.array([-0.9, 0.15, 0.002]),
        np.array([-1.1, 0.08, 0.001]),
    ]
    
    for sp in start_points:
        try:
            result = minimize(objective, sp, method='Nelder-Mead', 
                             options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
            if result.fun < best_bound:
                best_bound = result.fun
                best_c = result.x
        except Exception:
            pass
    
    # Also try a grid-based local search around the best point
    for scale in [0.1, 0.01, 0.001]:
        for i in range(-5, 6):
            for j in range(-5, 6):
                for k in range(-5, 6):
                    trial = best_c + np.array([i, j, k]) * scale
                    try:
                        b = _compute_bound(*trial)
                        if b < best_bound:
                            best_bound = b
                            best_c = trial
                    except Exception:
                        pass
    
    return np.array([best_c[0], best_c[1], best_c[2]])
# EVOLVE-BLOCK-END
