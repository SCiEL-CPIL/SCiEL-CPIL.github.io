"""Refined Cohn-Goncalves Laguerre construction: direct root optimization."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 600.0
_START = time.time()


def _time_left():
    return _DEADLINE - time.time()


def _laguerre_coeffs(n, alpha=1):
    """Coefficients of L_n^(alpha)(x) in ascending power of x."""
    coeff = np.zeros(n + 1)
    for k in range(n + 1):
        try:
            log_val = (math.lgamma(n + alpha + 1) - math.lgamma(n - k + 1)
                       - math.lgamma(k + alpha + 1))
            binom_val = math.exp(log_val)
        except Exception:
            binom_val = 1.0
        coeff[k] = ((-1) ** k) * binom_val / math.factorial(k)
    return coeff


def _compute_norm_sq(poly):
    """||psi||^2 = integral_0^inf psi(x)^2 x^1 e^{-x} dx  (exact)."""
    # psi(x) = sum p_k x^k
    # psi(x)^2 = sum c_j x^j, where c_j = sum_{i=0}^j p_i p_{j-i}
    # Integral = sum c_j * integral x^{j+1} e^{-x} dx = sum c_j * (j+1)!
    sq = np.convolve(poly, poly)
    max_k = len(sq) - 1
    # Weights are (j+1)! for j=0..max_k
    # j ranges from 0 to max_k. Term is x^{j+1}. Integral is (j+1)!.
    weights = np.array([math.factorial(j + 1) for j in range(max_k + 1)], dtype=float)
    return float(np.dot(sq, weights))


def _build_poly(roots, m=12):
    """psi(x) = L_m^1(x) * prod_i (1 - x/r_i)^2, as ascending coefficients."""
    coeff_L = _laguerre_coeffs(m, 1)
    poly = coeff_L.copy()
    for r in roots:
        # (1 - x/r)^2 = 1 - 2x/r + x^2/r^2
        # Coeffs: [1, -2/r, 1/r^2]
        q = np.array([1.0, -2.0 / r, 1.0 / (r * r)])
        poly = np.convolve(poly, q)
    return poly


def _c4_bound(roots, m=12):
    """Upper bound on C4 for given positive distinct roots (lower is better)."""
    if _time_left() < 1:
        return float('inf')
    if len(roots) != 12:
        return float('inf')
    if any(r <= 0 for r in roots):
        return float('inf')
    # Check distinctness with some tolerance
    if len(set(round(float(r), 8) for r in roots)) != 12:
        return float('inf')
    
    poly = _build_poly(roots, m)
    if not np.all(np.isfinite(poly)):
        return float('inf')
    
    norm_sq = _compute_norm_sq(poly)
    if not np.isfinite(norm_sq) or norm_sq <= 0:
        return float('inf')
    
    return 1.0 / norm_sq


def _objective(v, m=12):
    """Objective to minimize: -log(norm_sq). v = [log(r_1), ..., log(r_12)]."""
    if _time_left() < 0.5:
        return 1e100
    
    # Convert from log-space to root-space
    try:
        roots = np.exp(v)
    except OverflowError:
        return 1e100
        
    if not np.all(np.isfinite(roots)):
        return 1e100
    if np.any(roots <= 0) or np.any(roots > 1e15):
        return 1e100
        
    # Check distinctness
    if len(set(np.round(roots, 8))) != 12:
        return 1e100

    poly = _build_poly(roots, m)
    if not np.all(np.isfinite(poly)):
        return 1e100
        
    norm_sq = _compute_norm_sq(poly)
    if not np.isfinite(norm_sq) or norm_sq <= 0:
        return 1e100
        
    return -math.log(norm_sq)


def construct():
    m = 12
    
    # Initial guess: Geometric roots starting at 4, ratio 2
    # This is the "seed" from the previous iteration
    r0, q = 4.0, 2.0
    seed_roots = [r0 * (q ** i) for i in range(12)]
    seed_val = _c4_bound(seed_roots, m)
    if not np.isfinite(seed_val):
        seed_val = 1.0
    best_roots = seed_roots[:]
    best_val = seed_val
    
    # Optimization bounds in log-space
    # Roots should be positive. 
    # Lower bound: not too close to 0 to avoid numerical issues with 1/r^2
    # Upper bound: not too large to avoid overflow in coefficients
    # log(0.1) ~ -2.3, log(1000) ~ 6.9
    bounds = [(-4.0, 8.0) for _ in range(12)]
    
    # Multi-start strategy
    # 1. The seed point
    # 2. Random perturbations of the seed
    # 3. Random starts within bounds
    
    starts = []
    
    # Seed in log-space
    starts.append(np.log(seed_roots))
    
    # Deterministic random starts
    rng = np.random.default_rng(42)
    
    # Generate 50 random starts
    for _ in range(50):
        # Sample log(roots) uniformly in a reasonable range
        # Let's try a wide range: log(0.5) to log(50)
        l_roots = rng.uniform(math.log(0.5), math.log(50.0), size=12)
        starts.append(l_roots)
        
    # Also try some structured starts:
    # Geometric with different ratios
    for q_test in [0.5, 0.8, 1.2, 1.5, 3.0, 5.0]:
        for r0_test in [1.0, 2.0, 5.0, 10.0]:
            roots_test = [r0_test * (q_test ** i) for i in range(12)]
            if all(0.1 < r < 100 for r in roots_test):
                starts.append(np.log(roots_test))
                
    # Run optimization
    for x0 in starts:
        if _time_left() < 5:
            break
            
        try:
            res = minimize(
                _objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
        except Exception:
            continue
            
        if not res.success and res.fun > 1e99:
            continue
            
        try:
            cand_roots = np.exp(res.x)
        except OverflowError:
            continue
            
        if len(cand_roots) != 12 or np.any(cand_roots <= 0) or np.any(cand_roots > 1e15):
            continue
        if len(set(np.round(cand_roots, 8))) != 12:
            continue
            
        cv = _c4_bound(cand_roots, m)
        if np.isfinite(cv) and cv < best_val:
            best_val = cv
            best_roots = cand_roots[:]
            
            # If we found a significantly better solution, refine it with higher precision
            if _time_left() > 10:
                try:
                    res_refine = minimize(
                        _objective, 
                        np.log(best_roots), 
                        method='L-BFGS-B', 
                        bounds=bounds,
                        options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-10}
                    )
                    cand_roots_ref = np.exp(res_refine.x)
                    if (len(cand_roots_ref) == 12 and np.all(cand_roots_ref > 0) 
                        and np.all(cand_roots_ref < 1e15) 
                        and len(set(np.round(cand_roots_ref, 8))) == 12):
                        cv_ref = _c4_bound(cand_roots_ref, m)
                        if np.isfinite(cv_ref) and cv_ref < best_val:
                            best_val = cv_ref
                            best_roots = cand_roots_ref[:]
                except Exception:
                    pass

    # Safety check
    if len(best_roots) != 12 or any(r <= 0 for r in best_roots):
        best_roots = seed_roots[:]
    if len(set(round(r, 8) for r in best_roots)) != 12:
        best_roots = seed_roots[:]
        
    return [float(r) for r in best_roots]
# EVOLVE-BLOCK-END
