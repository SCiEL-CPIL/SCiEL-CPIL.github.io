"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _get_hermite_coeffs(n):
    """Get coefficients of physicist's Hermite polynomial Hn (highest power first)."""
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([2.0, 0.0])
    h_prev2 = np.array([1.0])
    h_prev1 = np.array([2.0, 0.0])
    for k in range(2, n + 1):
        term1 = np.zeros(len(h_prev1) + 1)
        term1[:-1] = 2.0 * h_prev1
        term2 = 2.0 * (k - 1) * h_prev2
        if len(term2) < len(term1):
            term2 = np.pad(term2, (0, len(term1) - len(term2)))
        elif len(term2) > len(term1):
            term1 = np.pad(term1, (0, len(term2) - len(term1)))
        h_curr = term1 - term2
        h_prev2, h_prev1 = h_prev1, h_curr
    return h_prev1

# Precompute Hermite coefficients
_H0 = _get_hermite_coeffs(0)
_H4 = _get_hermite_coeffs(4)
_H8 = _get_hermite_coeffs(8)
_H12 = _get_hermite_coeffs(12)

# Pad to same length
_MAX_LEN = max(len(_H0), len(_H4), len(_H8), len(_H12))
def _pad(c):
    return np.pad(c, (0, _MAX_LEN - len(c)))

_H0p = _pad(_H0)
_H4p = _pad(_H4)
_H8p = _pad(_H8)
_H12p = _pad(_H12)

# Hn(0) is the constant term (last element)
_H0_0 = _H0p[-1]
_H4_0 = _H4p[-1]
_H8_0 = _H8p[-1]
_H12_0 = _H12p[-1]

def _compute_value(c0, c1, c2):
    """Compute the value for given coefficients using precomputed data."""
    c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    g = c0 * _H0p + c1 * _H4p + c2 * _H8p + c3 * _H12p
    
    # q = g / x^2
    if len(g) < 3:
        return float('inf')
    if abs(g[-1]) > 1e-10 or abs(g[-2]) > 1e-10:
        return float('inf')
    q = g[:-2]
    
    # Remove leading zeros
    while len(q) > 1 and abs(q[0]) < 1e-14:
        q = q[1:]
    if len(q) == 0:
        return float('inf')
    
    roots = np.roots(q)
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    
    # Check sign change at each real root
    sign_changing = []
    for r in real_roots:
        eps = 1e-8 * (1 + abs(r))
        val_before = np.polyval(q, r - eps)
        val_after = np.polyval(q, r + eps)
        if val_before * val_after < 0:
            sign_changing.append(r)
    
    if len(sign_changing) == 0:
        return float('inf')
    
    r = max(sign_changing)
    return r**2 / (2 * np.pi)

def _smooth_objective(coeffs):
    """Smooth surrogate using log-sum-exp over sign-changing roots."""
    c0, c1, c2 = coeffs
    c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    g = c0 * _H0p + c1 * _H4p + c2 * _H8p + c3 * _H12p
    
    if len(g) < 3:
        return float('inf')
    if abs(g[-1]) > 1e-10 or abs(g[-2]) > 1e-10:
        return float('inf')
    q = g[:-2]
    
    while len(q) > 1 and abs(q[0]) < 1e-14:
        q = q[1:]
    if len(q) == 0:
        return float('inf')
    
    roots = np.roots(q)
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    
    sign_changing = []
    for r in real_roots:
        eps = 1e-8 * (1 + abs(r))
        val_before = np.polyval(q, r - eps)
        val_after = np.polyval(q, r + eps)
        if val_before * val_after < 0:
            sign_changing.append(r)
    
    if len(sign_changing) == 0:
        return float('inf')
    
    r_squared = np.array(sign_changing)**2
    max_r2 = np.max(r_squared)
    if max_r2 > 1e10:
        return float('inf')
    log_sum = max_r2 + np.log(np.sum(np.exp(r_squared - max_r2)))
    return log_sum / (2 * np.pi)

def construct():
    start_time = time.time()
    deadline = start_time + 180
    
    seed = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_value = _compute_value(*seed)
    best_coeffs = seed.copy()
    
    # Phase 1: Coarse search with differential evolution
    elapsed = time.time() - start_time
    if elapsed < 100:
        try:
            bounds = [(-5, 5), (-2, 2), (-0.1, 0.1)]
            result = differential_evolution(
                _smooth_objective, bounds, maxiter=50, tol=1e-6,
                popsize=15, seed=42, polish=False, updating='immediate'
            )
            elapsed = time.time() - start_time
            if result.fun < best_value:
                test_val = _compute_value(*result.x)
                if test_val < best_value:
                    best_value = test_val
                    best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Local refinement with Nelder-Mead
    elapsed = time.time() - start_time
    if elapsed < 160:
        try:
            result = minimize(
                _smooth_objective, best_coeffs, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            elapsed = time.time() - start_time
            if result.fun < best_value:
                test_val = _compute_value(*result.x)
                if test_val < best_value:
                    best_value = test_val
                    best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final polish with exact objective
    elapsed = time.time() - start_time
    if elapsed < 190:
        try:
            result = minimize(
                _compute_value, best_coeffs, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if result.fun < best_value:
                best_value = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
