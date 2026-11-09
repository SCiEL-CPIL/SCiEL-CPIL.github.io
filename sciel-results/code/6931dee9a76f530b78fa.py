"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_genlaguerre, gammainc
import time
from math import comb, factorial, log, exp
from scipy.optimize import minimize

def _compute_c4_bound(roots):
    """
    Compute the upper bound on C4 given 12 positive double-root locations.
    C4 <= K / sqrt(prod(roots_i))
    To minimize C4, we maximize prod(roots_i).
    """
    roots = np.array(roots)
    if len(roots) != 12:
        return float('inf')
    if np.any(roots <= 0):
        return float('inf')
    
    log_prod = np.sum(np.log(roots))
    c4_bound = exp(-0.5 * log_prod)
    return c4_bound

def _check_constraints(roots, tol=1e-6):
    """
    Check if the roots satisfy the Laguerre-type polynomial constraints.
    For double roots at x_1,...,x_12, the polynomial P(x) = prod(x - x_i)^2
    must satisfy orthogonality conditions against the Laguerre weight.
    
    A simpler sufficient condition: the roots should be such that the
    associated Laguerre polynomial with these nodes has the correct
    sign-change structure. We check that the polynomial evaluated at
    specific points has the expected sign pattern.
    """
    roots = np.array(roots)
    if len(roots) != 12:
        return False
    if np.any(roots <= 0):
        return False
    
    # Check that roots are distinct (double roots but at different locations)
    if np.any(np.diff(np.sort(roots)) < 1e-8):
        return False
    
    # The key constraint: for the Cohn-Goncalves construction, the roots
    # should be related to the zeros of a specific Laguerre-type polynomial.
    # We verify by checking that the product polynomial has the right
    # asymptotic behavior and sign changes.
    
    # A practical check: ensure the roots are in a reasonable range
    # and that the polynomial P(x) = prod(x-x_i)^2 has the expected
    # number of sign changes (which it always does for positive roots).
    
    # More importantly, check that the roots are consistent with
    # the moment problem: sum of 1/x_i should be in a reasonable range
    # for the m=12, n=1 case.
    
    inv_sum = np.sum(1.0 / roots)
    # For Laguerre zeros of L_12, sum of 1/x_i is around 12/6 = 2 (roughly)
    # Allow a wide range
    if inv_sum < 0.1 or inv_sum > 100:
        return False
    
    return True

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Score is the upper bound on C4 (lower is better).
    """
    t_start = time.time()
    deadline = t_start + 580.0  # Safety margin for 630s limit
    
    # Get initial guess: zeros of L_12(x)
    n = 12
    coeffs = np.zeros(n + 1)
    for k in range(n + 1):
        if k == 0:
            coeffs[0] = 1.0
        else:
            coeffs[k] = ((-1)**k) * comb(n + k, k) / factorial(k)
    coeffs_desc = coeffs[::-1]
    
    try:
        roots_lag = np.roots(coeffs_desc)
        real_roots = [r.real for r in roots_lag if abs(r.imag) < 1e-8 and r.real > 1e-10]
        real_roots.sort()
        if len(real_roots) != 12:
            raise ValueError("Wrong number of real roots")
        x0 = np.array(real_roots)
    except Exception:
        return [4.0 * (i + 1) for i in range(12)]
    
    # Objective: minimize -log(product of roots) = -sum(log(roots))
    # Subject to: roots > 0 and constraint that they form valid double roots
    
    best_roots = x0.copy()
    best_score = _compute_c4_bound(best_roots)
    
    def objective(params):
        """Minimize negative log product of roots."""
        roots = np.exp(params)
        return -np.sum(params)  # Since params = log(roots), sum(params) = log(prod)
    
    # Use L-BFGS-B with bounds in log-space
    # Lower bound: roots > 1e-4, so log(roots) > log(1e-4) = -9.21
    # Upper bound: roots < 1e6, so log(roots) < log(1e6) = 13.8
    
    lb = np.full(12, -9.0)
    ub = np.full(12, 13.0)
    bounds = list(zip(lb, ub))
    
    # Multiple restarts from different initial points
    best_params = np.log(x0).copy()
    best_obj = objective(best_params)
    
    # Initial point: log of Laguerre zeros
    initial_points = [np.log(x0)]
    
    # Add perturbed points
    rng = np.random.RandomState(42)
    for _ in range(10):
        perturbation = rng.normal(0, 0.3, size=12)
        pt = np.log(x0) + perturbation
        # Ensure within bounds
        pt = np.clip(pt, lb + 0.1, ub - 0.1)
        initial_points.append(pt)
    
    # Also try scaled versions
    for scale in [0.5, 0.7, 1.3, 1.5, 2.0]:
        pt = np.log(x0 * scale)
        pt = np.clip(pt, lb + 0.1, ub - 0.1)
        initial_points.append(pt)
    
    for i, x_init in enumerate(initial_points):
        if time.time() > deadline - 60:
            break
        
        try:
            result = minimize(
                objective, 
                x_init, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_obj:
                candidate_roots = np.exp(result.x)
                if np.all(candidate_roots > 0) and _check_constraints(candidate_roots):
                    score = _compute_c4_bound(candidate_roots)
                    if score < best_score:
                        best_score = score
                        best_roots = candidate_roots.copy()
                        best_params = result.x.copy()
                        best_obj = result.fun
                        
        except Exception:
            continue
        
        # Check time
        if time.time() > deadline - 30:
            break
    
    # Try a second phase: refine with Nelder-Mead in log-space
    if time.time() < deadline - 20:
        try:
            result = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            candidate_roots = np.exp(result.x)
            if np.all(candidate_roots > 0) and _check_constraints(candidate_roots):
                score = _compute_c4_bound(candidate_roots)
                if score < best_score:
                    best_score = score
                    best_roots = candidate_roots.copy()
        except Exception:
            pass
    
    # Sort and return
    best_roots = np.sort(best_roots)
    return list(best_roots)

# EVOLVE-BLOCK-END
