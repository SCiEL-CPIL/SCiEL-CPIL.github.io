"""Spectral moment approach: minimize uncertainty bound via direct numerical integration
and trust-region reflective optimization with adaptive starting points."""

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import time

_DEADLINE = time.time() + 160


def _make_g_coefficients(a0, a4, a8):
    """
    Given free coefficients a0, a4, a8, compute all polynomial coefficients
    P(x) = sum_{k=0}^{6} p_k x^{2k} for g(x) = P(x) e^{-x^2/2}.
    a12 is fixed to enforce g(0) = 0.
    """
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    p0 = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    p1 = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p2 = a4 + 210.0 * a8 + 13860.0 * a12
    p3 = -28.0 * a8 - 6930.0 * a12
    p4 = a8 + 1155.0 * a12
    p5 = -66.0 * a12
    p6 = a12
    return np.array([p0, p1, p2, p3, p4, p5, p6]), a12


def _integrate_moments(p, order):
    """
    Compute m_k = integral x^{2k} |P(x)|^2 e^{-x^2} dx for k=0,1,2
    using adaptive quadrature.
    """
    def integrand_k(x, k):
        val = 0.0
        for i in range(7):
            for j in range(7):
                val += p[i] * p[j] * x**(2*(i+j))
        return x**(2*k) * val * np.exp(-x**2)
    
    results = []
    for k in range(order):
        val, err = quad(integrand_k, 0, 50, args=(k,), limit=200, epsabs=1e-12, epsrel=1e-12)
        results.append(2.0 * val)  # even function, integrate 0 to inf
    return np.array(results)


def _integrate_fourier_moments(p, order):
    """
    Compute n_k = 2*pi * integral xi^{2k} |Q(xi)|^2 e^{-xi^2} dxi for k=0,1,2
    where Q(xi) = sum p_k (-1)^k xi^{2k}
    """
    q = np.array([p[k] * ((-1.0)**k) for k in range(7)])
    
    def integrand_k(xi, k):
        val = 0.0
        for i in range(7):
            for j in range(7):
                val += q[i] * q[j] * xi**(2*(i+j))
        return 2.0 * np.pi * xi**(2*k) * val * np.exp(-xi**2)
    
    results = []
    for k in range(order):
        val, err = quad(integrand_k, 0, 50, args=(k,), limit=200, epsabs=1e-12, epsrel=1e-12)
        results.append(2.0 * val)
    return np.array(results)


def _bound_value(params):
    """Compute the uncertainty bound for given parameters."""
    p, _ = _make_g_coefficients(params[0], params[1], params[2])
    
    m = _integrate_moments(p, 3)
    n = _integrate_fourier_moments(p, 3)
    
    if m[0] <= 0 or n[0] <= 0 or not np.all(np.isfinite(m)) or not np.all(np.isfinite(n)):
        return 1e18
    
    vf = m[1] / m[0]
    vF = n[1] / n[0]
    
    if vf <= 0 or vF <= 0:
        return 1e18
    
    return np.sqrt(vf * vF)


def _bound_grad(params):
    """Compute bound and numerical gradient."""
    eps = 1e-7
    val = _bound_value(params)
    
    grad = np.zeros(3)
    for i in range(3):
        params_plus = params.copy()
        params_plus[i] += eps
        val_plus = _bound_value(params_plus)
        grad[i] = (val_plus - val) / eps
    
    return val, grad


def construct():
    """
    Find optimal Hermite coefficients to minimize the uncertainty bound.
    Returns array of three coefficients [a0, a4, a8].
    """
    # Seed starting point
    seed = np.array([-1.13, 0.04, 0.000308642])
    
    # Generate diverse starting points
    rng = np.random.default_rng(2024)
    starts = [seed]
    
    # Structured perturbations
    for scale in [0.1, 0.5, 1.0, 2.0]:
        for _ in range(5):
            starts.append(seed + rng.normal(0, scale, 3))
    
    # Random starts in reasonable range
    for _ in range(15):
        starts.append(rng.uniform(-3, 3, 3))
    
    best_val = _bound_value(seed)
    best_params = seed.copy()
    
    # Track for trust-region reflective
    for i, start in enumerate(starts):
        if time.time() > _DEADLINE - 30:
            break
        
        try:
            # First try L-BFGS-B with numerical gradient
            res = minimize(
                _bound_value, start,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 5000}
            )
            
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
        
        # Try trust-region reflective from best so far
        if i % 3 == 0 and time.time() < _DEADLINE - 40:
            try:
                res2 = minimize(
                    _bound_value, best_params,
                    method='TNC',
                    options={'maxiter': 200, 'ftol': 1e-14}
                )
                if res2.fun < best_val and np.isfinite(res2.fun):
                    best_val = res2.fun
                    best_params = res2.x.copy()
            except Exception:
                pass
    
    # Final high-precision polish with Nelder-Mead
    if time.time() < _DEADLINE - 15:
        try:
            res = minimize(
                _bound_value, best_params,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-16, 'adaptive': True}
            )
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # One more L-BFGS-B polish
    if time.time() < _DEADLINE - 10:
        try:
            res = minimize(
                _bound_value, best_params,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 20000}
            )
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None or not np.all(np.isfinite(best_params)):
        return seed
    
    return np.array(best_params)
