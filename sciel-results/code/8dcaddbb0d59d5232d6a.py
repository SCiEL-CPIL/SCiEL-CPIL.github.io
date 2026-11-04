"""Refined Cohn-Goncalves Laguerre construction: 12-dim log-space DE + multi-start L-BFGS-B."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

_START = time.time()
_DEADLINE = 600.0
_MARGIN = 30.0

def _time_left():
    return _DEADLINE - (time.time() - _START) - _MARGIN

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Returns a list of 12 positive root locations.
    Score is the upper bound on C4 (lower is better for the evaluator's raw value,
    but the search score is seed/value, so higher search score = better).
    """
    
    m = 12
    
    # Precompute the transformation matrix from standard basis to Laguerre basis
    A = np.zeros((m + 1, m + 1))
    for k in range(m + 1):
        for j in range(k + 1):
            try:
                A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
            except Exception:
                A[k, j] = 0.0
                
    M = A.T
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        M_inv = np.linalg.pinv(M)
    
    # Precompute weights for norm computation
    weights = np.arange(m + 1) + 1

    def compute_norm_sq(roots):
        """
        Compute the squared norm of the polynomial in the Laguerre basis.
        roots: array of 12 positive numbers.
        Uses stable polynomial construction via iterative multiplication.
        """
        # Build polynomial coefficients in standard basis (ascending)
        # P(x) = prod(x - r_i)
        # Start with p = [1] (constant polynomial 1)
        p = np.array([1.0])
        
        for r in roots:
            # Multiply p by (x - r): new_p[j] = p[j-1] - r*p[j]
            # In ascending order: if p = [c0, c1, ..., cd], then (x-r)*p = [c0*(-r), c1*(-r)+c0, c2*(-r)+c1, ..., cd*(-r)]
            new_p = np.zeros(len(p) + 1)
            new_p[0] = -r * p[0]
            new_p[1:-1] = p[:-1] - r * p[1:]
            new_p[-1] = p[-1]
            p = new_p
        
        # Ensure p is length m+1
        if len(p) < m + 1:
            p = np.pad(p, (0, m + 1 - len(p)))
        elif len(p) > m + 1:
            p = p[:m+1]
            
        # c = M_inv @ p
        c = M_inv @ p
        
        # Norm squared: ||P||^2 = sum c_k^2 * (k+1)
        norm_sq = np.sum(c**2 * weights)
        
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_func(log_roots):
        """
        Minimize the upper bound on C4.
        Bound ~ 1 / ||P||^2.
        So we minimize 1/norm_sq, which is equivalent to maximizing norm_sq.
        We return 1/norm_sq.
        """
        roots = np.exp(log_roots)
        
        # Check for numerical issues
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
            
        norm_sq = compute_norm_sq(roots)
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    # Optimization Strategy:
    # 1. Global search over 12 log-root parameters using Differential Evolution.
    # 2. Multi-start local refinement with L-BFGS-B from diverse initial points.
    
    # Seed: geometric sequence a=4, q=2
    seed_log_roots = np.log(4.0) + np.arange(12) * np.log(2.0)
    
    # Bounds for log_roots
    bounds = [(-4.0, 10.0)] * 12
    
    best_roots = None
    best_val = 1e10
    best_log_roots = None
    
    # Phase 1: Differential Evolution for global search
    if _time_left() > 90:
        try:
            result = differential_evolution(
                cost_func, 
                bounds, 
                seed=42, 
                maxiter=150, 
                popsize=12, 
                tol=1e-8, 
                polish=True,
                init='sobol',
                updating='deferred',
                workers=1
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x
                best_roots = np.exp(best_log_roots)
                
        except Exception:
            pass
    
    # Fallback / Initial guess if DE fails or to ensure we have a baseline
    seed_roots = 4.0 * (2.0 ** np.arange(12))
    seed_val = cost_func(seed_log_roots)
    
    if best_roots is None or seed_val < best_val:
        best_val = seed_val
        best_roots = seed_roots
        best_log_roots = seed_log_roots

    # Phase 2: Multi-start local refinement with L-BFGS-B
    # Generate diverse starting points for local refinement
    start_points = [best_log_roots.copy()]
    
    # Add perturbed versions of the best point
    rng = np.random.RandomState(42)
    for _ in range(5):
        perturbation = rng.normal(0, 0.5, size=12)
        perturbed = best_log_roots + perturbation
        # Clip to bounds
        perturbed = np.clip(perturbed, -4.0, 10.0)
        start_points.append(perturbed)
    
    # Add a few geometric sequences with different parameters
    for a, q in [(2.0, 1.5), (6.0, 1.8), (1.0, 2.5)]:
        geo_log_roots = np.log(a) + np.arange(12) * np.log(q)
        geo_log_roots = np.clip(geo_log_roots, -4.0, 10.0)
        start_points.append(geo_log_roots)
    
    for x0 in start_points:
        if _time_left() < 20:
            break
        try:
            res = minimize(
                cost_func,
                x0=x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_log_roots = res.x
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass
            
    # Phase 3: Final high-precision polish
    if _time_left() > 15 and best_log_roots is not None:
        try:
            res = minimize(
                cost_func,
                x0=best_log_roots,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_log_roots = res.x
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass
            
    # Ensure roots are positive
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(12)]
