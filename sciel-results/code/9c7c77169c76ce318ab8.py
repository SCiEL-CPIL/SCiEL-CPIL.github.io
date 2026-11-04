import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_laguerre
import time

def _check_distinctness(roots, min_gap=1e-6):
    """Check if roots are distinct and sorted."""
    if len(roots) == 0:
        return True
    roots_sorted = np.sort(roots)
    if np.any(roots_sorted <= 0):
        return False
    if np.any(np.isnan(roots_sorted)):
        return False
    diffs = np.diff(roots_sorted)
    if np.any(diffs < min_gap):
        return False
    return True

def _compute_c4_bound_fast(roots):
    """Compute the C4 upper bound for given double-root positions using Gauss-Laguerre quadrature."""
    roots = np.asarray(roots, dtype=np.float64)
    
    # Strict distinctness check
    if not _check_distinctness(roots, min_gap=1e-8):
        return np.inf
    
    n_quad = 200
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # Vectorized: compute log_P for all roots at once
    diff = x_quad[:, np.newaxis] - roots[np.newaxis, :]
    
    # Check for zero differences in quadrature points (should not happen with distinct roots and Laguerre nodes, but safe)
    # Actually, Laguerre nodes are distinct. Roots are distinct. So diff should not be 0.
    # But if a root is exactly a node, diff is 0. log(0) is -inf.
    # We add epsilon to avoid -inf, but if diff is 0, the integral is actually singular?
    # No, P(x) = prod (x-r_i)^2. If x=r_i, P(x)=0.
    # The integral is finite. The log calculation might fail if diff is exactly 0.
    # np.log(0) is -inf. sum of -inf is -inf. exp(-inf) is 0.
    # So P=0 at nodes where root exists. This is correct.
    # However, if two roots are very close, numerical issues might arise.
    
    log_abs_diff = np.log(np.abs(diff) + 1e-300)
    log_P = 2.0 * np.sum(log_abs_diff, axis=1)
    
    P = np.exp(log_P)
    P2 = P * P
    
    int_P = np.sum(w_quad * P)
    int_P2 = np.sum(w_quad * P2)
    
    if int_P <= 0 or np.isinf(int_P) or np.isinf(int_P2) or np.isnan(int_P) or np.isnan(int_P2):
        return np.inf
    
    c4 = int_P2 / (int_P * int_P)
    
    if np.isnan(c4) or np.isinf(c4):
        return np.inf
        
    return c4

def _objective_log(log_roots):
    """Objective in log-space for better constraint handling."""
    roots = np.exp(log_roots)
    return _compute_c4_bound_fast(roots)

def _get_equilibrium_roots():
    """Get roots from the corrected mean-field equilibrium condition."""
    m = 12
    # Start from Laguerre roots as initial guess
    x0, _ = roots_laguerre(m)
    roots = x0.copy()
    
    # Use Newton-Raphson to solve sum_{j!=i} 2/(r_i - r_j) = 1
    # This is the equilibrium for Dyson gas with linear potential.
    
    for iteration in range(500):
        new_roots = roots.copy()
        for i in range(m):
            s = 0.0
            ds = 0.0
            for j in range(m):
                if j != i:
                    d = roots[i] - roots[j]
                    if abs(d) < 1e-10:
                        # Roots too close, perturb slightly to avoid division by zero
                        # This indicates instability
                        return roots # Return current best effort or fail
                    s += 2.0 / d
                    ds -= 2.0 / (d * d)
            
            if abs(ds) > 1e-15:
                new_roots[i] = roots[i] - (s - 1.0) / ds
            else:
                new_roots[i] = roots[i]
        
        if np.max(np.abs(new_roots - roots)) < 1e-12:
            roots = new_roots
            break
        roots = new_roots
        
        # Check for distinctness during iteration
        if not _check_distinctness(roots, min_gap=1e-6):
            # If they collide, break and return last good state or Laguerre roots
            break
    
    roots = np.sort(roots)
    if _check_distinctness(roots, min_gap=1e-6):
        return roots
    else:
        # Fallback to Laguerre roots which are distinct
        return x0

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s kill
    
    # Get initial guesses from different strategies
    laguerre_roots, _ = roots_laguerre(12)
    laguerre_roots = np.sort(laguerre_roots)
    
    equilibrium_roots = _get_equilibrium_roots()
    if not _check_distinctness(equilibrium_roots, min_gap=1e-6):
        equilibrium_roots = laguerre_roots.copy()
        
    best_roots = equilibrium_roots.copy()
    best_score = _compute_c4_bound_fast(best_roots)
    
    # Also try Laguerre roots
    lag_score = _compute_c4_bound_fast(laguerre_roots)
    if lag_score < best_score and _check_distinctness(laguerre_roots, min_gap=1e-6):
        best_score = lag_score
        best_roots = laguerre_roots.copy()
    
    if time.time() > deadline - 120:
        if _check_distinctness(best_roots, min_gap=1e-6):
            return [float(xi) for xi in best_roots]
        else:
            # Return Laguerre roots as they are always distinct
            return [float(xi) for xi in laguerre_roots]
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial points
    initial_points = [equilibrium_roots.copy(), laguerre_roots.copy()]
    
    # Perturbed versions
    for scale in [0.05, 0.1, 0.2, 0.3, 0.5]:
        perturbed = best_roots * (1 + scale * rng.standard_normal(12))
        perturbed = np.maximum(perturbed, 1e-3)
        perturbed = np.sort(perturbed)
        # Ensure distinctness before adding to list
        if _check_distinctness(perturbed, min_gap=1e-5):
            initial_points.append(perturbed)
    
    # Scaled versions
    for scale in [0.8, 1.2, 1.5, 2.0]:
        scaled = best_roots * scale
        if _check_distinctness(scaled, min_gap=1e-5):
            initial_points.append(scaled.copy())
    
    # Random initial points
    for _ in range(5):
        rand_roots = rng.uniform(0.1, 50, size=12)
        rand_roots = np.sort(rand_roots)
        if _check_distinctness(rand_roots, min_gap=1e-5):
            initial_points.append(rand_roots)
    
    # Optimize in log-space for better constraint handling
    bounds_log = [(-20, 20)] * 12  # log(1e-9) to log(1e9)
    
    for x_init in initial_points:
        if time.time() > deadline - 60:
            break
        
        try:
            x_init_log = np.log(np.maximum(x_init, 1e-10))
            
            result = minimize(
                _objective_log,
                x_init_log,
                method='L-BFGS-B',
                bounds=bounds_log,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if result.success or np.isfinite(result.fun):
                candidate = np.exp(result.x)
                # CRITICAL: Check distinctness
                if _check_distinctness(candidate, min_gap=1e-6):
                    candidate = np.sort(candidate)
                    score = _compute_c4_bound_fast(candidate)
                    if score < best_score and np.isfinite(score):
                        best_score = score
                        best_roots = candidate.copy()
        except Exception:
            pass
    
    # Final polish in log-space with tighter tolerance
    if time.time() < deadline - 10:
        try:
            if _check_distinctness(best_roots, min_gap=1e-6):
                best_roots_log = np.log(best_roots)
                result_final = minimize(
                    _objective_log,
                    best_roots_log,
                    method='L-BFGS-B',
                    bounds=bounds_log,
                    options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14}
                )
                if np.isfinite(result_final.fun):
                    candidate = np.exp(result_final.x)
                    if _check_distinctness(candidate, min_gap=1e-6):
                        candidate = np.sort(candidate)
                        score = _compute_c4_bound_fast(candidate)
                        if score < best_score and np.isfinite(score):
                            best_score = score
                            best_roots = candidate.copy()
        except Exception:
            pass
    
    # Final safety check
    if _check_distinctness(best_roots, min_gap=1e-6):
        return [float(xi) for xi in best_roots]
    else:
        # Fallback to Laguerre roots
        return [float(xi) for xi in laguerre_roots]
