"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, brentq
import time


# Hermite polynomial coefficients (physicists' convention): H_n(x) = sum_k a[k] x^(n-2k)
# H_0 = 1
# H_4 = 16x^4 - 48x^2 + 12
# H_8 = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 112
# H_12 = 4096x^12 - 46080x^10 + 302400x^8 - 806400x^6 + 1008000x^4 - 604800x^2 + 10368

# g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
# g(0) = c0*1 + c1*12 + c2*112 + c3*10368 = 0
# => c3 = -(c0 + 12*c1 + 112*c2) / 10368

# q(x) = g(x) / x^2. Since g(0)=0 and g is even, g(x) = x^2 * q(x) where q is even polynomial of degree 10.
# q(x) = a12*x^10 + a10*x^8 + a8*x^6 + a6*x^4 + a4*x^2 + a2
# where:
# a12 = 4096*c3
# a10 = -46080*c3 + 256*c2
# a8  = 302400*c3 - 1344*c2 - 48*c1
# a6  = -806400*c3 + 2688*c2 - 16*c1 + c0
# a4  = 1008000*c3 - 1344*c2
# a2  = -604800*c3 + 112*c2 + 12*c1

def _build_q_coeffs(c0, c1, c2):
    c3 = -(c0 + 12.0 * c1 + 112.0 * c2) / 10368.0
    a12 = 4096.0 * c3
    a10 = -46080.0 * c3 + 256.0 * c2
    a8  = 302400.0 * c3 - 1344.0 * c2 - 48.0 * c1
    a6  = -806400.0 * c3 + 2688.0 * c2 - 16.0 * c1 + c0
    a4  = 1008000.0 * c3 - 1344.0 * c2
    a2  = -604800.0 * c3 + 112.0 * c2 + 12.0 * c1
    return np.array([a12, a10, a8, a6, a4, a2])


def _q_eval(coeffs, x):
    """Evaluate q(x) = a12*x^10 + a10*x^8 + a8*x^6 + a6*x^4 + a4*x^2 + a2"""
    # Use Horner's method in z = x^2
    z = x * x
    # q in terms of z: a12*z^5 + a10*z^4 + a8*z^3 + a6*z^2 + a4*z + a2
    return ((a12 * z + a10) * z + a8) * z * z + (a6 * z + a4) * z + a2


def _q_eval_vec(coeffs, x):
    """Vectorized evaluation of q(x)."""
    z = x * x
    a12, a10, a8, a6, a4, a2 = coeffs
    return ((a12 * z + a10) * z + a8) * z * z + (a6 * z + a4) * z + a2


def _find_largest_positive_sign_change_root(coeffs):
    """Find the largest positive x where q(x) changes sign (crosses zero).
    We look for the largest positive simple root of q.
    """
    a12, a10, a8, a6, a4, a2 = coeffs
    
    # Check leading coefficient sign for behavior at infinity
    # If a12 < 0, q -> -inf as x -> inf
    # If a12 > 0, q -> +inf as x -> inf
    
    # We need the largest positive root where q changes sign.
    # Scan from large x downward to find sign changes.
    
    # Start from a large value and scan downward
    # The roots are bounded; for a degree-10 polynomial, use a reasonable upper bound
    # |root| <= 1 + max(|a_i|/|a_12|) if a12 != 0
    
    if abs(a12) < 1e-30:
        # Degenerate: check lower degree
        # Find effective degree
        pass
    
    # Use a grid scan to find sign change intervals, then refine with brentq
    # Upper bound for roots: Cauchy bound
    if abs(a12) > 1e-30:
        cauchy_bound = 1.0 + max(abs(a10), abs(a8), abs(a6), abs(a4), abs(a2)) / abs(a12)
    else:
        cauchy_bound = 100.0
    
    cauchy_bound = min(cauchy_bound, 1e6)  # Safety cap
    
    # Scan on a grid to find sign changes
    n_scan = 2000
    x_max = cauchy_bound * 1.1
    x_grid = np.linspace(1e-10, x_max, n_scan)
    q_grid = _q_eval_vec(coeffs, x_grid)
    
    # Find intervals where sign changes
    roots = []
    for i in range(n_scan - 1):
        if q_grid[i] == 0:
            roots.append(x_grid[i])
        elif q_grid[i] * q_grid[i+1] < 0:
            # Sign change: use brentq to refine
            try:
                root = brentq(lambda x: _q_eval(coeffs, x), x_grid[i], x_grid[i+1], 
                              xtol=1e-14, rtol=1e-14, maxiter=100)
                roots.append(root)
            except Exception:
                pass
    
    if not roots:
        return None
    
    # We want the largest positive root where q changes sign
    # All roots found via sign change are sign-changing by construction
    return max(roots)


def _objective(params):
    """Objective: r^2 / (2*pi) where r is the largest positive sign-changing root of q.
    Lower is better.
    """
    c0, c1, c2 = params
    coeffs = _build_q_coeffs(c0, c1, c2)
    r = _find_largest_positive_sign_change_root(coeffs)
    if r is None:
        return 1e10  # Penalty
    return r * r / (2.0 * np.pi)


def construct():
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin before 210s kill
    
    best_value = None
    best_params = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val = _objective(best_params)
    
    if best_val is None or best_val > 1e9:
        best_val = 1e10
    
    # Multi-start Nelder-Mead search
    rng = np.random.RandomState(42)
    
    # Initial candidates: seed + random perturbations
    starts = [best_params.copy()]
    
    # Add random starts in a reasonable range
    for _ in range(30):
        p = best_params + rng.normal(0, 0.5, size=3)
        starts.append(p)
    
    # Also try some structured starts
    for scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
        for _ in range(5):
            p = best_params + scale * rng.normal(0, 1.0, size=3)
            starts.append(p)
    
    # Sort by initial objective
    starts_with_obj = []
    for s in starts:
        v = _objective(s)
        starts_with_obj.append((v, s))
    starts_with_obj.sort(key=lambda x: x[0])
    
    # Run Nelder-Mead from top starts
    for i, (init_val, init_params) in enumerate(starts_with_obj[:15]):
        if time.time() > deadline - 30.0:
            break
        
        try:
            result = minimize(_objective, init_params, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14,
                                     'adaptive': True})
            
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except Exception:
            continue
    
    # High-precision polish: run more iterations from the best point
    if time.time() < deadline - 10.0:
        try:
            result = minimize(_objective, best_params, method='Nelder-Mead',
                             options={'maxiter': 20000, 'xatol': 1e-15, 'fatol': 1e-16,
                                     'adaptive': True})
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Final verification
    final_val = _objective(best_params)
    if final_val < best_val:
        best_val = final_val
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
