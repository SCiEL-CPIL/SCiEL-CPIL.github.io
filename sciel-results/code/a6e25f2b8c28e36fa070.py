"""Refined Cohn-Goncalves Laguerre-polynomial construction for the Fourier
sign uncertainty upper bound.

Structural change: instead of using the zeros of a Laguerre polynomial
(which minimizes a different criterion), we directly minimize the
weight-integrated squared product ∫₀^∞ [∏(x-a_i)²] e^{-x} dx over the
12 positive double-root locations {a_i}.  This is the actual quantity
that appears in the Cohn-Gonçalves Fourier sign uncertainty bound.

We use the Laguerre zeros as a warm start and refine with scipy.optimize.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import roots_laguerre, gamma
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 630.0
_SAFETY = 30.0
_M = 12


def _deadline_exceeded():
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY)


def _compute_score(roots):
    """Compute the weight-integrated squared product ∫₀^∞ [∏(x-a_i)²] e^{-x} dx.
    
    This integral can be computed analytically using the fact that
    ∫₀^∞ x^k e^{-x} dx = k!
    
    [∏(x-a_i)²] = ∏(x² - 2a_i x + a_i²)
    
    We expand this polynomial and integrate term by term.
    """
    roots = np.asarray(roots, dtype=float)
    m = len(roots)
    
    # Build the polynomial ∏(x - a_i)^2 = ∏(x^2 - 2a_i x + a_i^2)
    # This is a polynomial of degree 2m
    # Coefficients in descending order of power
    # Start with 1
    coeffs = np.array([1.0])
    for a in roots:
        # Multiply by (x^2 - 2a*x + a^2)
        new_coeffs = np.zeros(len(coeffs) + 2)
        for j, c in enumerate(coeffs):
            new_coeffs[j] += c  # x^2 * c*x^j = c*x^(j+2)
            new_coeffs[j+1] += -2*a*c  # -2a*x * c*x^j = -2a*c*x^(j+1)
            new_coeffs[j+2] += a*a*c  # a^2 * c*x^j = a^2*c*x^j
        coeffs = new_coeffs
    
    # coeffs[0] is coefficient of x^(2m), coeffs[2m] is constant term
    # ∫₀^∞ x^k e^{-x} dx = k!
    # So the integral = sum_{k=0}^{2m} coeffs[k] * (2m - k)!
    degree = 2 * m
    total = 0.0
    for k in range(degree + 1):
        # coeffs[k] is coefficient of x^(degree - k)
        power = degree - k
        total += coeffs[k] * gamma(power + 1)
    return total


def _compute_score_fast(roots):
    """Faster computation using log-space to avoid overflow for large factorials."""
    roots = np.asarray(roots, dtype=float)
    m = len(roots)
    degree = 2 * m
    
    # Build polynomial coefficients
    coeffs = np.array([1.0])
    for a in roots:
        new_coeffs = np.zeros(len(coeffs) + 2)
        for j, c in enumerate(coeffs):
            new_coeffs[j] += c
            new_coeffs[j+1] += -2*a*c
            new_coeffs[j+2] += a*a*c
        coeffs = new_coeffs
    
    # Compute sum coeffs[k] * (degree-k)!
    # Use log for stability
    total = 0.0
    for k in range(degree + 1):
        power = degree - k
        # (power)! = gamma(power+1)
        log_term = np.log(abs(coeffs[k])) + np.log(gamma(power + 1)) if coeffs[k] != 0 else -np.inf
        if coeffs[k] < 0:
            total -= np.exp(log_term)
        else:
            total += np.exp(log_term)
    return total


def _objective(params):
    """Objective function for optimization: the weight-integrated squared product."""
    roots = np.sort(params)
    if np.any(roots <= 0):
        return 1e300
    if np.any(np.diff(roots) < 1e-10):
        return 1e300
    try:
        return _compute_score_fast(roots)
    except Exception:
        return 1e300


def _gradient(params):
    """Numerical gradient."""
    eps = 1e-7
    grad = np.zeros_like(params)
    f0 = _objective(params)
    for i in range(len(params)):
        params_p = params.copy()
        params_p[i] += eps
        grad[i] = (_objective(params_p) - f0) / eps
    return grad


def construct():
    """Return a list of 12 positive double-root locations for the Laguerre
    combination (m=12, n=1).  Score (upper bound on C4) is the raw value of
    the resulting construction; lower is better.
    """
    best = None
    best_score = np.inf
    
    try:
        # Warm start: zeros of L_12
        zeros, _ = roots_laguerre(_M)
        zeros = np.sort(zeros)
        
        # Verify
        if np.all(zeros > 0) and np.all(np.diff(zeros) > 0):
            score = _compute_score_fast(zeros)
            best = [float(z) for z in zeros]
            best_score = score
        
        # Deadline check
        if _deadline_exceeded():
            return best
        
        # Optimize: minimize the weight-integrated squared product
        # Use L-BFGS-B with bounds to ensure positivity
        x0 = zeros.copy()
        bounds = [(1e-6, None)] * _M
        
        result = minimize(
            _objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        
        if result.success or result.fun < best_score:
            opt_roots = np.sort(result.x)
            if np.all(opt_roots > 0) and np.all(np.diff(opt_roots) > 1e-10):
                opt_score = _compute_score_fast(opt_roots)
                if opt_score < best_score:
                    best = [float(r) for r in opt_roots]
                    best_score = opt_score
        
        # Deadline check
        if _deadline_exceeded():
            return best
        
        # Try a second optimization from a different starting point
        # Start from a slightly perturbed version
        x0_2 = zeros * (1 + 0.1 * np.random.RandomState(42).rand(_M))
        x0_2 = np.sort(x0_2)
        
        result2 = minimize(
            _objective,
            x0_2,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        
        if result2.success or result2.fun < best_score:
            opt_roots2 = np.sort(result2.x)
            if np.all(opt_roots2 > 0) and np.all(np.diff(opt_roots2) > 1e-10):
                opt_score2 = _compute_score_fast(opt_roots2)
                if opt_score2 < best_score:
                    best = [float(r) for r in opt_roots2]
                    best_score = opt_score2
        
        return best
        
    except Exception:
        if best is not None:
            return best
        return [4.0 * (i + 1) for i in range(_M)]
# EVOLVE-BLOCK-END
