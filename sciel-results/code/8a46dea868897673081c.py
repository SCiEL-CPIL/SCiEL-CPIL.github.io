"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, brentq
from scipy.special import roots_laguerre
import time

def _compute_c4_bound_fast(roots):
    """Compute the C4 upper bound for given double-root positions using Gauss-Laguerre quadrature."""
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf
    
    n_quad = 200
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # Vectorized: compute log_P for all roots at once
    diff = x_quad[:, np.newaxis] - roots[np.newaxis, :]
    log_abs_diff = np.log(np.abs(diff) + 1e-300)
    log_P = 2.0 * np.sum(log_abs_diff, axis=1)
    
    P = np.exp(log_P)
    P2 = P * P
    
    int_P = np.sum(w_quad * P)
    int_P2 = np.sum(w_quad * P2)
    
    if int_P <= 0 or np.isinf(int_P) or np.isinf(int_P2):
        return np.inf
    
    c4 = int_P2 / (int_P * int_P)
    return c4

def _compute_c4_bound(roots):
    """Alias for compatibility."""
    return _compute_c4_bound_fast(roots)

def _objective(roots):
    """Objective function for optimization."""
    return _compute_c4_bound_fast(roots)

def _objective_from_log(log_roots):
    """Objective in log-space to ensure positivity."""
    roots = np.exp(log_roots)
    return _compute_c4_bound_fast(roots)

def _pole_equilibrium_residual(roots, i):
    """
    Compute the equilibrium residual for root i.
    At optimal configuration, the ratio of integrals over adjacent intervals
    should satisfy a specific relation derived from the Laguerre polynomial structure.
    
    For a product P(x) = prod_j (x - r_j)^2, the optimal double-root positions
    satisfy a fixed-point condition where each root balances the influence
    of its neighbors through the weight function e^{-x}.
    """
    roots = np.asarray(roots, dtype=np.float64)
    n = len(roots)
    
    # The equilibrium condition: at optimum, d/dr_i [log(int P^2 w / (int P w)^2)] = 0
    # This simplifies to a condition involving the ratio of integrals over
    # intervals [r_i, r_{i+1}] and [r_{i-1}, r_i]
    
    # Numerical gradient of log(C4) with respect to r_i
    eps = 1e-7 * max(1.0, abs(roots[i]))
    
    r_plus = roots.copy()
    r_plus[i] += eps
    r_minus = roots.copy()
    r_minus[i] -= eps
    
    # Ensure positivity
    if r_minus[i] <= 0:
        r_minus[i] = eps * 0.5
    
    c4_plus = _compute_c4_bound_fast(r_plus)
    c4_minus = _compute_c4_bound_fast(r_minus)
    c4_center = _compute_c4_bound_fast(roots)
    
    if np.isinf(c4_plus) or np.isinf(c4_minus) or np.isinf(c4_center):
        return 0.0
    
    # Central difference for log(C4)
    grad = (np.log(c4_plus) - np.log(c4_minus)) / (2.0 * eps)
    return grad

def _fixed_point_iteration(roots, max_iter=100, tol=1e-10):
    """
    Apply fixed-point iteration to find the equilibrium root configuration.
    Each root is updated based on the local gradient of log(C4).
    """
    roots = np.asarray(roots, dtype=np.float64).copy()
    n = len(roots)
    
    for it in range(max_iter):
        max_change = 0.0
        for i in range(n):
            grad = _pole_equilibrium_residual(roots, i)
            if np.isnan(grad) or np.isinf(grad):
                continue
            
            # Newton-like step with adaptive damping
            # Approximate second derivative for step size
            eps = 1e-5 * max(1.0, abs(roots[i]))
            r_plus = roots.copy()
            r_plus[i] += eps
            c4_plus = _compute_c4_bound_fast(r_plus)
            
            if np.isinf(c4_plus) or c4_plus <= 0:
                continue
            
            c4_center = _compute_c4_bound_fast(roots)
            if np.isinf(c4_center) or c4_center <= 0:
                continue
            
            # Approximate curvature
            c4_minus = _compute_c4_bound_fast(roots - np.eye(n)[i] * eps)
            if np.isinf(c4_minus) or c4_minus <= 0:
                continue
            
            log_c4_plus = np.log(c4_plus)
            log_c4_center = np.log(c4_center)
            log_c4_minus = np.log(c4_minus)
            
            d2 = (log_c4_plus - 2 * log_c4_center + log_c4_minus) / (eps * eps)
            if d2 <= 0:
                d2 = 1e-8  # Small positive default
            
            step = -grad / d2
            # Limit step size
            max_step = 0.5 * abs(roots[i])
            step = np.clip(step, -max_step, max_step)
            
            new_val = roots[i] + step
            if new_val > 0:
                change = abs(new_val - roots[i])
                roots[i] = new_val
                max_change = max(max_change, change)
        
        if max_change < tol:
            break
    
    # Sort and return
    return np.sort(roots)

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    deadline = time.time() + 580  # Safety margin before 630s kill
    
    # Start from standard Laguerre roots as initial guess
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    
    best_roots = x0.copy()
    best_score = _compute_c4_bound_fast(best_roots)
    
    rng = np.random.default_rng(12345)
    
    # Strategy 1: Geometric parameterization in log-space
    def _geometric_objective(params):
        a, b = params
        if b <= 0:
            return 1e10
        roots = np.exp(a + b * np.arange(12))
        if np.any(roots <= 0) or np.any(roots > 1e6) or np.any(roots < 1e-6):
            return 1e10
        return _compute_c4_bound_fast(roots)
    
    # Multistart over geometric parameterization
    for a_init in np.linspace(-2, 2, 15):
        for b_init in np.linspace(0.1, 1.0, 15):
            if time.time() > deadline - 60:
                break
            try:
                result = minimize(
                    _geometric_objective,
                    np.array([a_init, b_init]),
                    method='L-BFGS-B',
                    bounds=[(-5, 5), (0.05, 2.0)],
                    options={'maxiter': 150, 'ftol': 1e-15, 'gtol': 1e-13}
                )
                if result.fun < best_score:
                    a, b = result.x
                    roots = np.exp(a + b * np.arange(12))
                    if np.all(roots > 0) and np.all(roots < 1e6):
                        candidate = np.sort(roots)
                        score = _compute_c4_bound_fast(candidate)
                        if score < best_score:
                            best_score = score
                            best_roots = candidate.copy()
            except Exception:
                pass
        if time.time() > deadline - 60:
            break
    
    # Strategy 2: Fixed-point iteration from multiple starting points
    starting_points = [
        x0.copy(),  # Laguerre roots
        np.exp(np.linspace(np.log(0.5), np.log(20), 12)),  # Geometric
        np.exp(np.linspace(np.log(0.3), np.log(30), 12)),  # Wider geometric
        np.exp(np.linspace(np.log(1.0), np.log(10), 12)),  # Tighter geometric
    ]
    
    # Add perturbed versions
    for _ in range(5):
        perturbed = x0 * np.exp(rng.standard_normal(12) * 0.3)
        starting_points.append(np.sort(perturbed))
    
    for start in starting_points:
        if time.time() > deadline - 90:
            break
        
        try:
            # First, refine with gradient descent
            result = minimize(
                _objective,
                start,
                method='L-BFGS-B',
                bounds=[(1e-8, None)] * 12,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if result.fun < best_score:
                candidate = np.sort(result.x)
                if np.all(candidate > 0):
                    score = _compute_c4_bound_fast(candidate)
                    if score < best_score:
                        best_score = score
                        best_roots = candidate.copy()
            
            # Then apply fixed-point iteration from the refined point
            if time.time() < deadline - 60:
                fp_roots = _fixed_point_iteration(best_roots, max_iter=50, tol=1e-9)
                if np.all(fp_roots > 0) and np.all(np.isfinite(fp_roots)):
                    score = _compute_c4_bound_fast(fp_roots)
                    if score < best_score:
                        best_score = score
                        best_roots = fp_roots.copy()
        except Exception:
            pass
    
    # Strategy 3: Final refinement with tighter tolerances
    if time.time() < deadline - 30:
        try:
            result_final = minimize(
                _objective,
                best_roots,
                method='L-BFGS-B',
                bounds=[(1e-8, None)] * 12,
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if result_final.fun < best_score:
                candidate = np.sort(result_final.x)
                if np.all(candidate > 0):
                    score = _compute_c4_bound_fast(candidate)
                    if score < best_score:
                        best_score = score
                        best_roots = candidate.copy()
        except Exception:
            pass
    
    # Strategy 4: One more fixed-point pass from the best
    if time.time() < deadline - 20:
        try:
            fp_roots = _fixed_point_iteration(best_roots, max_iter=80, tol=1e-10)
            if np.all(fp_roots > 0) and np.all(np.isfinite(fp_roots)):
                score = _compute_c4_bound_fast(fp_roots)
                if score < best_score:
                    best_score = score
                    best_roots = fp_roots.copy()
        except Exception:
            pass
    
    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
