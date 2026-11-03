"""Refined Cohn-Goncalves Laguerre construction: direct root optimization."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_DEADLINE = 600.0
_START = time.time()


def _time_left():
    return _DEADLINE - time.time()


def _laguerre_coeffs(n, alpha=1):
    """Compute coefficients of L_n^(alpha)(x) in ascending power of x."""
    coeff = np.zeros(n + 1)
    for k in range(n + 1):
        try:
            from math import lgamma
            log_val = lgamma(n + alpha + 1) - lgamma(n - k + 1) - lgamma(k + alpha + 1)
            binom_val = math.exp(log_val)
        except Exception:
            binom_val = 1.0
        coeff[k] = ((-1) ** k) * binom_val / math.factorial(k)
    return coeff


def _compute_integral(poly):
    """Compute integral_0^inf psi(x)^2 x^1 e^{-x} dx for psi with ascending coeffs."""
    sq = np.convolve(poly, poly)
    max_k = len(sq) - 1
    # weights[k] = (k+1)!
    weights = np.ones(max_k + 1)
    for j in range(1, max_k + 1):
        weights[j] = weights[j - 1] * (j + 1)
    return np.dot(sq, weights)


def _c4_bound(roots, m=12):
    """Upper bound on C4 = 1 / integral."""
    if _time_left() < 5:
        return float('inf')
    
    coeff_L = _laguerre_coeffs(m, 1)
    poly = coeff_L.copy()
    for r in roots:
        if r <= 0:
            return float('inf')
        q = np.array([r * r, -2.0 * r, 1.0])
        poly = np.convolve(poly, q)
    
    if not np.all(np.isfinite(poly)):
        return float('inf')
    
    val = _compute_integral(poly)
    if not np.isfinite(val) or val <= 0:
        return float('inf')
    return 1.0 / val


def _integral_value(roots, m=12):
    """Return the integral value (to maximize)."""
    if _time_left() < 5:
        return 0.0
    c4 = _c4_bound(roots, m)
    if not np.isfinite(c4):
        return 0.0
    return 1.0 / c4


def _nelder_mead_maximize(func, x0, bounds_low, bounds_high, max_iter=500):
    """Simple Nelder-Mead to maximize func. Returns best x and value."""
    n = len(x0)
    # Initialize simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        x = np.array(x0, dtype=float)
        step = 0.1 * (bounds_high[i] - bounds_low[i]) if bounds_high[i] > bounds_low[i] else 1.0
        x[i] += step
        x = np.clip(x, bounds_low, bounds_high)
        simplex.append(x)
    
    func_vals = [func(x) for x in simplex]
    best_x = x0[:]
    best_val = func(best_x)
    
    for it in range(max_iter):
        if _time_left() < 30:
            break
        # Sort by function value (descending for maximization)
        order = np.argsort(func_vals)[::-1]
        simplex = [simplex[i] for i in order]
        func_vals = [func_vals[i] for i in order]
        
        if func_vals[0] > best_val:
            best_val = func_vals[0]
            best_x = simplex[0].copy()
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + (centroid - simplex[-1])
        xr = np.clip(xr, bounds_low, bounds_high)
        fr = func(xr)
        
        if fr >= func_vals[0]:
            # Expansion
            xe = centroid + 2.0 * (centroid - simplex[-1])
            xe = np.clip(xe, bounds_low, bounds_high)
            fe = func(xe)
            if fe > fr:
                simplex[-1] = xe
                func_vals[-1] = fe
            else:
                simplex[-1] = xr
                func_vals[-1] = fr
        elif fr >= func_vals[-2]:
            simplex[-1] = xr
            func_vals[-1] = fr
        else:
            # Contraction
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            xc = np.clip(xc, bounds_low, bounds_high)
            fc = func(xc)
            if fc > func_vals[-1]:
                simplex[-1] = xc
                func_vals[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], bounds_low, bounds_high)
                    func_vals[i] = func(simplex[i])
    
    return best_x, best_val


def construct():
    m = 12
    n_roots = 12
    
    # Seed roots
    seed_roots = [4.0 * (i + 1) for i in range(n_roots)]
    seed_val = _c4_bound(seed_roots, m)
    if not np.isfinite(seed_val):
        seed_val = float('inf')
    
    best_roots = seed_roots[:]
    best_val = seed_val
    
    # Phase 1: Coarse grid over parametric family to find good region
    scales = np.linspace(1.0, 20.0, 30)
    alphas = np.linspace(0.5, 1.5, 20)
    
    for s in scales:
        if _time_left() < 120:
            break
        for a in alphas:
            if _time_left() < 60:
                break
            roots = [s * ((i + 1) ** a) for i in range(n_roots)]
            if min(roots) <= 0:
                continue
            val = _c4_bound(roots, m)
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_roots = roots[:]
    
    # Phase 2: Direct optimization of all 12 roots using Nelder-Mead
    if _time_left() > 60:
        # Objective: minimize C4 (maximize integral)
        def objective(roots_arr):
            if np.any(roots_arr <= 0):
                return float('inf')
            return _c4_bound(roots_arr, m)
        
        x0 = np.array(best_roots, dtype=float)
        # Bounds: roots should be positive, reasonable range
        bounds_low = np.ones(n_roots) * 0.1
        bounds_high = np.ones(n_roots) * 100.0
        
        # Run Nelder-Mead
        best_x, best_f = _nelder_mead_maximize(
            lambda x: -objective(x),  # negate for maximization
            x0, bounds_low, bounds_high, max_iter=300
        )
        
        # Convert back: we minimized -(-C4) = C4, so best_f = -C4
        # Actually objective returns C4, we maximize -C4, so best_f = -C4
        if np.isfinite(best_f):
            c4_candidate = -best_f
            if c4_candidate < best_val and np.all(best_x > 0):
                best_val = c4_candidate
                best_roots = list(best_x)
    
    # Phase 3: Fine-tune with smaller steps if time remains
    if _time_left() > 30 and best_val < seed_val:
        def objective2(roots_arr):
            if np.any(roots_arr <= 0):
                return float('inf')
            return _c4_bound(roots_arr, m)
        
        x0 = np.array(best_roots, dtype=float)
        bounds_low = np.ones(n_roots) * 0.05
        bounds_high = np.ones(n_roots) * 150.0
        
        # Multiple restarts with small perturbations
        for restart in range(5):
            if _time_left() < 20:
                break
            x0_pert = x0 + np.random.normal(0, 0.1 * np.max(x0), n_roots)
            x0_pert = np.clip(x0_pert, bounds_low, bounds_high)
            
            best_x, best_f = _nelder_mead_maximize(
                lambda x: -objective2(x),
                x0_pert, bounds_low, bounds_high, max_iter=100
            )
            
            if np.isfinite(best_f):
                c4_candidate = -best_f
                if c4_candidate < best_val and np.all(best_x > 0):
                    best_val = c4_candidate
                    best_roots = list(best_x)
                    x0 = best_x
    
    return best_roots
# EVOLVE-BLOCK-END
