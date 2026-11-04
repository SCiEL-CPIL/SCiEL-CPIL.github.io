"""Refined Cohn-Goncalves Laguerre construction optimizer (m=12, n=1)."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_genlaguerre, gammaln

_START = time.time()
_DEADLINE = 630.0 - 30.0  # safety margin

def _laguerre_norm_sq_from_roots(roots):
    """||P||^2 in the Laguerre inner product, P(x)=prod (x-r_i)^2.
    Use expansion P = sum_k c_k x^k, then <x^a,x^b> = Gamma(a+1) Gamma(b+1)."""
    # Build polynomial coefficients (highest degree first) from double roots
    # P(x) = prod_i (x - r_i)^2 = prod_i (x^2 - 2 r_i x + r_i^2)
    coeffs = np.array([1.0])  # will be [1,0,0,...] highest-first
    for r in roots:
        q = np.array([1.0, -2.0 * r, r * r])
        coeffs = np.convolve(coeffs, q)
    # coeffs[k] is coefficient of x^{(2m - k)}
    m = len(roots)
    deg = 2 * m
    # log Gamma for degrees 0..deg
    logg = gammaln(np.arange(deg + 1) + 1.0)
    # norm^2 = sum_{a,b} c_a c_b Gamma(a+1) Gamma(b+1) where c_a is coeff of x^a
    c = coeffs[::-1]  # now c[a] is coeff of x^a
    # weight matrix W_{a,b} = Gamma(a+1) Gamma(b+1) = exp(logg[a]+logg[b])
    # norm^2 = c^T W c = (c * exp(logg))^T (c * exp(logg)) ... no: W_{ab}=e^{logg a}e^{logg b}
    cw = c * np.exp(logg)
    return float(np.dot(cw, cw))

def _c4_from_roots(roots):
    """C4 upper bound = 1 / ||P||^2_Laguerre (smaller is better)."""
    n2 = _laguerre_norm_sq_from_roots(roots)
    if not np.isfinite(n2) or n2 <= 0:
        return 1e30
    return 1.0 / n2

def _objective(theta):
    """Objective in log-space: minimize C4 = 1/||P||^2."""
    if time.time() > _DEADLINE:
        return 1e30
    roots = np.exp(theta)
    # keep roots in a sane range to avoid overflow in convolution
    if np.any(roots > 1e6) or np.any(roots < 1e-6):
        return 1e30
    return _c4_from_roots(roots)

def construct():
    global _START
    # Seed: spread roots across a plausible range (Laguerre zeros of L_12 are ~[0.2, 38])
    # double roots -> use log-space init
    base = np.linspace(0.3, 30.0, 12)
    theta0 = np.log(base)

    best_roots = np.exp(theta0)
    best_val = _c4_from_roots(best_roots)

    # Multiple random restarts in log-space
    rng = np.random.default_rng(0)
    restarts = []
    for _ in range(6):
        spread = np.linspace(0.3, 30.0, 12)
        jitter = rng.uniform(-1.5, 1.5, 12)
        th = np.log(spread) + jitter
        restarts.append(th)
    restarts.insert(0, theta0.copy())

    for th in restarts:
        if time.time() > _DEADLINE - 20.0:
            break
        try:
            res = minimize(_objective, th, method='L-BFGS-B',
                           options={'maxiter': 400, 'maxfun': 8000,
                                    'ftol': 1e-12, 'gtol': 1e-8, 'disp': False})
        except Exception:
            continue
        if res.success or res.fun < best_val:
            cand_roots = np.exp(res.x)
            v = _c4_from_roots(cand_roots)
            if v < best_val and np.all(cand_roots > 0):
                best_val = v
                best_roots = cand_roots

    # Final high-precision polish pass on best roots
    if time.time() < _DEADLINE - 10.0:
        try:
            th_best = np.log(best_roots)
            res2 = minimize(_objective, th_best, method='L-BFGS-B',
                            options={'maxiter': 2000, 'maxfun': 40000,
                                     'ftol': 1e-15, 'gtol': 1e-10, 'disp': False})
            if res2.fun < best_val:
                cand_roots = np.exp(res2.x)
                v = _c4_from_roots(cand_roots)
                if v < best_val and np.all(cand_roots > 0):
                    best_val = v
                    best_roots = cand_roots
        except Exception:
            pass

    # Ensure strictly positive, sorted output of 12 double-root locations
    best_roots = np.asarray(best_roots, dtype=float)
    best_roots = np.clip(best_roots, 1e-9, 1e9)
    best_roots = np.sort(best_roots)
    return [float(x) for x in best_roots]
