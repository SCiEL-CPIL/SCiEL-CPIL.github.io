"""Uncertainty bound minimization via direct coefficient optimization with global search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from math import sqrt, pi
from itertools import product

_DEADLINE = 195.0


def _int_x2m(m):
    """∫_{-∞}^{∞} x^{2m} e^{-x^2} dx = sqrt(π) (2m-1)!! / 2^m"""
    if m == 0:
        return sqrt(pi)
    val = sqrt(pi)
    for j in range(1, m + 1):
        val *= (2 * j - 1) / 2.0
    return val


# Precompute Hermite polynomial coefficients
# H_0(x) = 1
# H_4(x) = 16x^4 - 48x^2 + 12
# H_8(x) = 256x^8 - 4480x^6 + 30240x^4 - 60480x^2 + 2520
# H_12(x) = 4096x^12 - 147456x^10 + 2177280x^8 - 14515200x^6 + 38976000x^4 - 39916800x^2 + 27720

# Coefficients as arrays indexed by power
_H0 = np.array([1.0])  # power 0
_H4 = np.array([0.0, 0.0, 0.0, 0.0, 16.0, 0.0, -48.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
_H8 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 256.0, 0.0, -4480.0, 0.0, 0.0])
_H8[4] = 30240.0
_H8[2] = -60480.0
_H8[0] = 2520.0
_H12 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
_H12[12] = 4096.0
_H12[10] = -147456.0
_H12[8] = 2177280.0
_H12[6] = -14515200.0
_H12[4] = 38976000.0
_H12[2] = -39916800.0
_H12[0] = 27720.0

# Rebuild properly
_H0 = np.zeros(13)
_H0[0] = 1.0

_H4 = np.zeros(13)
_H4[4] = 16.0
_H4[2] = -48.0
_H4[0] = 12.0

_H8 = np.zeros(13)
_H8[8] = 256.0
_H8[6] = -4480.0
_H8[4] = 30240.0
_H8[2] = -60480.0
_H8[0] = 2520.0

_H12 = np.zeros(13)
_H12[12] = 4096.0
_H12[10] = -147456.0
_H12[8] = 2177280.0
_H12[6] = -14515200.0
_H12[4] = 38976000.0
_H12[2] = -39916800.0
_H12[0] = 27720.0

_H_COEFFS = np.array([_H0, _H4, _H8, _H12])  # shape (4, 13)

# H values at zero
_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 2520.0
_H12_0 = 27720.0


def _compute_moments(a0, a4, a8):
    """Compute the uncertainty bound given coefficients a0, a4, a8."""
    a12 = -(a0 * _H0_0 + a4 * _H4_0 + a8 * _H8_0) / _H12_0
    
    # Build p_poly = a0*H0 + a4*H4 + a8*H8 + a12*H12
    coeffs = np.array([a0, a4, a8, a12])
    p_poly = coeffs @ _H_COEFFS  # shape (13,)
    
    # p(x)^2 as polynomial
    p2_poly = np.zeros(25)  # powers 0..24
    for i in range(13):
        if p_poly[i] != 0.0:
            for j in range(13):
                p2_poly[i + j] += p_poly[i] * p_poly[j]
    
    # M0 = ∫ p^2 e^{-x^2} dx
    M0 = 0.0
    for k in range(0, 25, 2):
        m = k // 2
        M0 += p2_poly[k] * _int_x2m(m)
    
    # M2 = ∫ x^2 p^2 e^{-x^2} dx
    M2 = 0.0
    for k in range(0, 25, 2):
        m_plus_1 = (k + 2) // 2
        M2 += p2_poly[k] * _int_x2m(m_plus_1)
    
    # g'(x) = (p' - x p) e^{-x^2/2}
    # p'(x)
    p_prime = np.zeros(12)
    for i in range(1, 13):
        p_prime[i - 1] = p_poly[i] * i
    
    # (p')^2
    pprime_sq = np.zeros(23)  # powers 0..22
    for i in range(12):
        if p_prime[i] != 0.0:
            for j in range(12):
                pprime_sq[i + j] += p_prime[i] * p_prime[j]
    
    int_pprime_sq = 0.0
    for k in range(0, 23, 2):
        m = k // 2
        int_pprime_sq += pprime_sq[k] * _int_x2m(m)
    
    # x*p(x)
    xp = np.zeros(14)
    for i in range(13):
        xp[i + 1] = p_poly[i]
    
    # x*p(x) * p'(x)
    xppp = np.zeros(25)  # powers 0..24
    for i in range(14):
        if xp[i] != 0.0:
            for j in range(12):
                xppp[i + j] += xp[i] * p_prime[j]
    
    int_xppp = 0.0
    for k in range(0, 25, 2):
        m = k // 2
        int_xppp += xppp[k] * _int_x2m(m)
    
    int_gprime_sq = int_pprime_sq - 2.0 * int_xppp + M2
    
    if M0 <= 1e-15 or int_gprime_sq <= 1e-15:
        return np.inf
    
    bound = np.sqrt(M2 * int_gprime_sq) / M0
    return bound


def _objective(params):
    """Objective function for optimization."""
    return _compute_moments(params[0], params[1], params[2])


def construct():
    """
    Find coefficients (a0, a4, a8) that minimize the uncertainty bound.
    Returns np.array([a0, a4, a8]).
    """
    start_time = time.time()
    
    # Known good starting point
    a0_lin = -113.0 / 100
    a4_lin = 1.0 / 25
    a8_lin = 1.0 / 3240
    
    best_x = np.array([a0_lin, a4_lin, a8_lin])
    best_bound = _compute_moments(a0_lin, a4_lin, a8_lin)
    
    # Generate diverse starting points
    rng = np.random.default_rng(12345)
    starts = [best_x.copy()]
    
    # Scale variations
    for scale in [0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 3.0, 5.0]:
        s = best_x * scale
        starts.append(s.copy())
    
    # Sign flips
    for s0 in [-1.0, 1.0]:
        for s4 in [-1.0, 1.0]:
            for s8 in [-1.0, 1.0]:
                s = np.array([s0 * abs(a0_lin), s4 * abs(a4_lin), s8 * abs(a8_lin)])
                starts.append(s.copy())
    
    # Random perturbations in log space
    for _ in range(50):
        log_scale = rng.normal(0, 1.5)
        sign0 = rng.choice([-1.0, 1.0])
        sign4 = rng.choice([-1.0, 1.0])
        sign8 = rng.choice([-1.0, 1.0])
        s = np.array([
            sign0 * abs(a0_lin) * np.exp(rng.normal(0, 0.8)),
            sign4 * abs(a4_lin) * np.exp(rng.normal(0, 0.8)),
            sign8 * abs(a8_lin) * np.exp(rng.normal(0, 0.8))
        ])
        starts.append(s)
    
    # More aggressive random starts
    for _ in range(30):
        s = np.array([
            rng.normal(0, 2.0),
            rng.normal(0, 0.5),
            rng.normal(0, 0.1)
        ])
        starts.append(s)
    
    # Try multiple optimization methods from each start
    for x_start in starts:
        if time.time() - start_time > _DEADLINE - 15:
            break
        
        # Method 1: L-BFGS-B
        try:
            result = minimize(
                _objective,
                x_start,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result.fun < best_bound:
                best_bound = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
        
        # Method 2: Nelder-Mead (only if time permits)
        if time.time() - start_time < _DEADLINE - 10:
            try:
                result = minimize(
                    _objective,
                    x_start,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                if result.fun < best_bound:
                    best_bound = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
    
    # Final refinement with tight tolerances
    if time.time() - start_time < _DEADLINE - 5:
        try:
            result = minimize(
                _objective,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            if result.fun < best_bound:
                best_bound = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    # Another Nelder-Mead refinement
    if time.time() - start_time < _DEADLINE - 2:
        try:
            result = minimize(
                _objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            if result.fun < best_bound:
                best_bound = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    return best_x
# EVOLVE-BLOCK-END
