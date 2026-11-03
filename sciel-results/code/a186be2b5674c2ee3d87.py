"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import factorial
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE_MARGIN = 15.0  # seconds to spare before the 210s kill

def _remaining():
    return _START_TIME + 210.0 - _DEADLINE_MARGIN - time.time()

# Pre-compute Hermite polynomial coefficients as dicts {power: coeff}
def _hermite_coeffs(n):
    """Return dict mapping power -> coefficient for physicist's Hermite polynomial H_n(x)."""
    if n % 2 != 0:
        return {}
    coeffs = {}
    n2 = n // 2
    for k in range(n2 + 1):
        sign = 1 if k % 2 == 0 else -1
        coeff = sign * (2 ** (n - 2 * k)) * float(factorial(n)) / (float(factorial(k)) * float(factorial(n - 2 * k)))
        power = n - 2 * k
        coeffs[power] = coeffs.get(power, 0.0) + coeff
    return coeffs

# Pre-compute the Hermite coefficient dicts
_H0 = _hermite_coeffs(0)
_H4 = _hermite_coeffs(4)
_H8 = _hermite_coeffs(8)
_H12 = _hermite_coeffs(12)

# Pre-compute H_n(0) values
_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 1680.0
_H12_0 = 665280.0

# Pre-compute the integral of x^n * exp(-x^2) over the real line for even n
def _int_even(n):
    """Integral of x^n * exp(-x^2) dx from -inf to inf, for even n."""
    if n % 2 != 0:
        return 0.0
    if n == 0:
        return np.sqrt(np.pi)
    double_fact = 1.0
    for k in range(1, n, 2):
        double_fact *= k
    return np.sqrt(np.pi) * double_fact / (2 ** (n / 2.0))

# Pre-compute integrals up to degree 24 (max degree of P^2 where P has degree 12)
_INTEGRALS = {}
for n in range(0, 25):
    if n % 2 == 0:
        _INTEGRALS[n] = _int_even(n)

def _poly_mul(a, b):
    """Multiply two polynomials represented as dicts {power: coeff}."""
    result = {}
    for p1, c1 in a.items():
        for p2, c2 in b.items():
            p = p1 + p2
            result[p] = result.get(p, 0.0) + c1 * c2
    return result

def _compute_upper_bound(c0, c4, c8):
    """
    Compute the product ||g||_2^2 * ||F(g)||_2^2 for
    g(x) = P(x) * exp(-x^2/2), where P = c0*H0 + c4*H4 + c8*H8 + c12*H12
    and c12 is chosen so P(0) = 0.
    """
    c12 = -(c0 * _H0_0 + c4 * _H4_0 + c8 * _H8_0) / _H12_0

    # Build P(x)
    P = {}
    for c, hc in [(c0, _H0), (c4, _H4), (c8, _H8), (c12, _H12)]:
        for p, coeff in hc.items():
            P[p] = P.get(p, 0.0) + c * coeff

    # Compute ||g||_2^2 = ∫ P(x)^2 e^{-x^2} dx
    P2 = _poly_mul(P, P)
    norm_sq = 0.0
    for power, coeff in P2.items():
        if power % 2 == 0 and power in _INTEGRALS:
            norm_sq += coeff * _INTEGRALS[power]

    # Compute ||F(g)||_2^2 = 2π ∫ |P(iξ)|^2 e^{-ξ^2} dξ
    # |P(iξ)|^2 = P(iξ) * P(-iξ) as a polynomial in ξ
    abs2_poly = {}
    for k, ak in P.items():
        for j, aj in P.items():
            power = k + j
            # i^k * (-i)^j = (-1)^j * i^{k+j}
            m = k + j
            mod4 = m % 4
            if mod4 == 0:
                i_pow = 1.0
            elif mod4 == 1:
                i_pow = 0.0  # imaginary, will cancel
                continue
            elif mod4 == 2:
                i_pow = -1.0
            else:
                i_pow = 0.0  # imaginary, will cancel
                continue
            
            coeff = ak * aj * ((-1) ** j) * i_pow
            abs2_poly[power] = abs2_poly.get(power, 0.0) + coeff

    norm_f_sq = 0.0
    for power, coeff in abs2_poly.items():
        if power % 2 == 0 and power in _INTEGRALS:
            norm_f_sq += coeff * _INTEGRALS[power]
    
    norm_f_sq *= 2 * np.pi

    return norm_sq * norm_f_sq

def construct():
    t_start = time.time()
    
    # Seed value from the prior state of the art
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best = seed.copy()
    best_val = _compute_upper_bound(best[0], best[1], best[2])
    
    # Use Nelder-Mead optimization with multiple starting points
    def objective(params):
        if time.time() > t_start + 210.0 - _DEADLINE_MARGIN:
            return 1e30
        return _compute_upper_bound(params[0], params[1], params[2])
    
    # Multiple starting points to explore the landscape
    start_points = [
        seed,
        np.array([-1.0, 0.04, 0.0003]),
        np.array([-0.5, 0.02, 0.0001]),
        np.array([-1.5, 0.08, 0.001]),
        np.array([-0.8, 0.03, 0.0002]),
    ]
    
    for sp in start_points:
        if time.time() > t_start + 210.0 - _DEADLINE_MARGIN:
            break
        
        try:
            result = minimize(
                objective,
                sp,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best = result.x.copy()
        except Exception:
            continue
    
    # Additional local refinement around the best point
    if time.time() < t_start + 210.0 - _DEADLINE_MARGIN:
        try:
            result = minimize(
                objective,
                best,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best = result.x.copy()
        except Exception:
            pass
    
    return best
# EVOLVE-BLOCK-END
