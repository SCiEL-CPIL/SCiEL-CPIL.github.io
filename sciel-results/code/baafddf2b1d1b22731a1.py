"""
Tune Hermite-polynomial coefficients to minimize the uncertainty bound C4.
We search over (a0, a4, a8) with a12 determined by g(0)=0, and minimize
the upper bound on A(f)*A(Fhat(f)) via a smooth surrogate / direct optimization.

Key insight: For g(x) = P(x)*exp(-x^2/2) where P is a polynomial, the Fourier
transform (with unitary convention) is Fhat(omega) = Q(omega)*exp(-omega^2/2)
where Q(omega) = (-i)^{deg} P(-i omega) ... actually for even polynomials and
the standard FT, we can compute the Fourier transform analytically.

For the uncertainty product, we need:
  A_x = (1/||g||_2^2) * integral x^2 |g(x)|^2 dx
  A_w = (1/||g||_2^2) * integral omega^2 |Fhat(omega)|^2 domega

By Plancherel (unitary FT): ||Fhat||_2 = ||g||_2
And: integral omega^2 |Fhat(omega)|^2 domega = integral |g'(x)|^2 dx  (with unitary FT convention)

So C4 = A_x * A_w = (integral x^2 |g|^2 / ||g||_2^2) * (integral |g'|^2 / ||g||_2^2)

For g(x) = P(x) * exp(-x^2/2):
  g'(x) = (P'(x) - x*P(x)) * exp(-x^2/2)

All integrals can be computed analytically using:
  integral x^{2n} exp(-x^2) dx = (2n)! / (4^n n!) * sqrt(pi)
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from math import factorial as fac
import time

# Deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 195.0  # 195s wall-clock, keep 15s margin


def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _even_moment(n):
    """integral x^(2n) exp(-x^2) dx from -inf to inf = (2n)!/(4^n n!) * sqrt(pi)"""
    if n == 0:
        return np.sqrt(np.pi)
    return fac(2 * n) / (4.0 ** n * fac(n)) * np.sqrt(np.pi)


# Precompute moments up to what we need
# P(x) has degree up to 12 (in x^2, so up to x^24 in P^2)
# P(x)^2 has degree up to 24, so we need moments up to n=12
# x^2 * P(x)^2 has degree up to 26, so we need moments up to n=13
# Q(x) = P'(x) - x*P(x) has degree up to 13 (odd), Q^2 has degree up to 26, moments up to n=13
_MOMENTS = [_even_moment(n) for n in range(15)]


def _c4_bound_fast(a0, a4, a8):
    """
    Compute the upper bound C4 = A(f) * A(Fhat(f)) for the test function
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    where a12 is chosen so g(0) = 0.
    
    C4 = (integral x^2 |g|^2 / integral |g|^2) * (integral |g'|^2 / integral |g|^2)
    """
    # a12 = -(a0 + 12*a4 + 10080*a8) / 103219200
    a12 = -(a0 + 12.0 * a4 + 10080.0 * a8) / 103219200.0
    
    # Build P(x) = sum p_{2k} x^{2k}
    # H0 = 1
    # H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 4608x^6 + 30240x^4 - 60480x^2 + 10080
    # H12 = 4096x^12 - 184320x^10 + 3932160x^8 - 42467520x^6 + 254016000x^4 - 765427200x^2 + 103219200
    
    p0 = 0.0  # by construction
    p2 = -48.0 * a4 - 60480.0 * a8 - 765427200.0 * a12
    p4 = 16.0 * a4 + 30240.0 * a8 + 254016000.0 * a12
    p6 = -4608.0 * a8 - 42467520.0 * a12
    p8 = 256.0 * a8 + 3932160.0 * a12
    p10 = -184320.0 * a12
    p12 = 4096.0 * a12
    
    p_coeffs = [p0, p2, p4, p6, p8, p10, p12]  # p_coeffs[k] = coefficient of x^(2k)
    
    # norm2 = integral P(x)^2 exp(-x^2) dx
    norm2 = 0.0
    for i in range(7):
        pi = p_coeffs[i]
        if pi == 0.0:
            continue
        for j in range(i, 7):
            pj = p_coeffs[j]
            if pj == 0.0:
                continue
            m = i + j
            if i == j:
                norm2 += pi * pj * _MOMENTS[m]
            else:
                norm2 += 2.0 * pi * pj * _MOMENTS[m]
    
    if norm2 < 1e-30:
        return 1e10
    
    # x2_norm2 = integral x^2 P(x)^2 exp(-x^2) dx
    x2_norm2 = 0.0
    for i in range(7):
        pi = p_coeffs[i]
        if pi == 0.0:
            continue
        for j in range(i, 7):
            pj = p_coeffs[j]
            if pj == 0.0:
                continue
            m = i + j + 1
            if i == j:
                x2_norm2 += pi * pj * _MOMENTS[m]
            else:
                x2_norm2 += 2.0 * pi * pj * _MOMENTS[m]
    
    # Q(x) = P'(x) - x*P(x)
    # P'(x) = sum 2k p_{2k} x^{2k-1} for k=1..6
    # x*P(x) = sum p_{2k} x^{2k+1} for k=0..6
    # Q(x) = (2p2 - p0)x + (4p4 - p2)x^3 + (6p6 - p4)x^5 + (8p8 - p6)x^7 + (10p10 - p8)x^9 + (12p12 - p10)x^11 - p12*x^13
    
    q_coeffs = {}  # power -> coefficient
    q_coeffs[1] = 2.0 * p2 - p0
    q_coeffs[3] = 4.0 * p4 - p2
    q_coeffs[5] = 6.0 * p6 - p4
    q_coeffs[7] = 8.0 * p8 - p6
    q_coeffs[9] = 10.0 * p10 - p8
    q_coeffs[11] = 12.0 * p12 - p10
    q_coeffs[13] = -p12
    
    # deriv_norm2 = integral Q(x)^2 exp(-x^2) dx
    deriv_norm2 = 0.0
    powers = sorted(q_coeffs.keys())
    for idx_i in range(len(powers)):
        qi = q_coeffs[powers[idx_i]]
        if qi == 0.0:
            continue
        for idx_j in range(idx_i, len(powers)):
            qj = q_coeffs[powers[idx_j]]
            if qj == 0.0:
                continue
            m = powers[idx_i] + powers[idx_j]  # even
            n = m // 2
            if idx_i == idx_j:
                deriv_norm2 += qi * qj * _MOMENTS[n]
            else:
                deriv_norm2 += 2.0 * qi * qj * _MOMENTS[n]
    
    C4 = (x2_norm2 / norm2) * (deriv_norm2 / norm2)
    
    return C4


def construct():
    """
    Return the three coefficients (a0, a4, a8) that minimize C4.
    a12 is determined by g(0)=0.
    """
    # Start from the seed values
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Evaluate seed
    seed_val = _c4_bound_fast(*x0)
    best_x = x0.copy()
    best_val = seed_val
    
    # Structural change: Use differential evolution for global search,
    # then refine with L-BFGS-B and Nelder-Mead.
    
    # Differential evolution for global exploration
    if _time_left() > 60:
        try:
            bounds = [(-10, 10), (-1, 1), (-0.01, 0.01)]
            de_result = differential_evolution(
                _c4_bound_fast,
                bounds,
                seed=42,
                maxiter=100,
                popsize=20,
                tol=1e-10,
                mutation=(0.5, 1.5),
                recombination=0.7,
                polish=False,
                init='sobol'
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_x = de_result.x
        except Exception:
            pass
    
    # Local refinement with L-BFGS-B
    if _time_left() > 30:
        try:
            res = minimize(
                _c4_bound_fast,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'gtol': 1e-12, 'ftol': 1e-14}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    # Final polish with Nelder-Mead
    if _time_left() > 15:
        try:
            res = minimize(
                _c4_bound_fast,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    # Additional multi-start refinement if time permits
    if _time_left() > 20:
        rng = np.random.RandomState(123)
        for _ in range(10):
            if _time_left() < 10:
                break
            pert = rng.randn(3) * [0.1, 0.01, 0.001]
            start = best_x + pert
            try:
                res = minimize(
                    _c4_bound_fast,
                    start,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
            except Exception:
                pass
    
    return np.array(best_x)
# EVOLVE-BLOCK-END
