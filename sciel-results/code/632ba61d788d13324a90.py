# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import hermite
from math import factorial, sqrt, pi

def _get_hermite_coeffs(n):
    """Return coefficients of H_n(x) in descending order."""
    h = hermite(n)
    return np.asarray(h.coef, dtype=float)

def _pad_coeffs(c, target_deg):
    """Pad polynomial coefficients (descending) to target degree."""
    current_deg = len(c) - 1
    zeros_needed = target_deg - current_deg
    if zeros_needed > 0:
        return np.concatenate([np.zeros(zeros_needed), c])
    return c

def _integrate_poly_gauss(poly_coeffs, a_val):
    """
    Compute integral of poly(x) * exp(-a_val * x^2) from -inf to inf.
    poly_coeffs are in descending order.
    """
    total = 0.0
    n_coeffs = len(poly_coeffs)
    for i, c in enumerate(poly_coeffs):
        power = (n_coeffs - 1) - i
        if power % 2 != 0:
            continue
        if c == 0:
            continue
        m = power // 2
        # (2m-1)!! = (2m)! / (2^m m!)
        double_fact = factorial(2*m) / (factorial(m) * (2**m))
        integral_val = c * sqrt(pi) * double_fact / ((2 * a_val) ** (m + 0.5))
        total += integral_val
    return total

def _bound_analytic(coeffs):
    a0, a1, a2 = coeffs
    # Calculate a3 such that P(0)=0
    H0_0, H4_0, H8_0, H12_0 = 1.0, -12.0, 1680.0, -2075040.0
    a3 = -(a0 * H0_0 + a1 * H4_0 + a2 * H8_0) / H12_0
    
    # Get polynomial coefficients for H0, H4, H8, H12
    c0 = _get_hermite_coeffs(0)
    c4 = _get_hermite_coeffs(4)
    c8 = _get_hermite_coeffs(8)
    c12 = _get_hermite_coeffs(12)
    
    max_deg = 12
    P = (a0 * _pad_coeffs(c0, max_deg) + 
         a1 * _pad_coeffs(c4, max_deg) + 
         a2 * _pad_coeffs(c8, max_deg) + 
         a3 * _pad_coeffs(c12, max_deg))
         
    # K = ∫ P(x)^2 e^{-x^2} dx
    P_sq = np.polymul(P, P)
    K = _integrate_poly_gauss(P_sq, 1.0)
    
    # J = ∫ P(x)^4 e^{-2x^2} dx
    P_four = np.polymul(np.polymul(P, P), np.polymul(P, P))
    J = _integrate_poly_gauss(P_four, 2.0)
    
    if K <= 0:
        return 1e18
    return J / K

def construct():
    # Multistart optimization of the three free coefficients.
    best = None
    best_val = np.inf
    rng = np.random.default_rng(0)
    starts = [
        [-113 / 100.0, 1 / 25.0, 1 / 3240.0],  # literature seed
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.5, 0.1],
        [0.5, -0.5, 0.2],
        [rng.normal(size=3) for _ in range(8)],
    ]
    for s in starts:
        s = np.asarray(s, dtype=float)
        try:
            res = minimize(
                _bound_analytic,
                s,
                method="Powell",
                options={"maxiter": 2000, "ftol": 1e-12},
            )
            val = _bound_analytic(res.x)
            if val < best_val:
                best_val = val
                best = res.x
        except Exception:
            continue
    if best is None:
        return np.array([-113 / 100.0, 1 / 25.0, 1 / 3240.0])
        
    # Refine with Nelder-Mead for local polishing
    try:
        res2 = minimize(
            _bound_analytic,
            best,
            method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 1e-10, "fatol": 1e-12},
        )
        if _bound_analytic(res2.x) < best_val:
            best = res2.x
    except Exception:
        pass
    return np.asarray(best, dtype=float)
# EVOLVE-BLOCK-END
