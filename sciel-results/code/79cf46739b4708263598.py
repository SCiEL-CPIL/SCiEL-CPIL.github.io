"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_genlaguerre, roots_genlaguerre
import time
import math

def construct():
    m = 12
    n = 1
    
    start_time = time.time()
    deadline = 600.0
    
    def time_ok():
        return (time.time() - start_time) < (deadline - 30.0)
    
    # Precompute the transformation matrix from standard basis to Laguerre basis
    # P(x) = sum c_k L_k^(1)(x)
    # L_k^(1)(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    
    A = np.zeros((m + 1, m + 1))
    for k in range(m + 1):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # Solve for c given p. c = (A^T)^{-1} p
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)
    
    # Precompute weights for norm calculation: w_k = k + 1 (for alpha=1)
    laguerre_weights = np.arange(m + 1) + 1.0
    
    def _compute_norm_sq(roots):
        """
        Compute the squared norm of the polynomial in the Laguerre basis.
        roots: array of 12 positive numbers.
        P(x) = prod_i (x - r_i)
        """
        x = np.asarray(roots, dtype=np.float64)
        if x.shape != (m,) or np.any(x <= 0) or np.any(np.isnan(x)) or np.any(np.isinf(x)):
            return 0.0
        
        # Build polynomial coefficients in standard basis (ascending)
        # np.poly returns descending coeffs: [1, c1, ..., cm]
        # We need ascending: [cm, ..., c1, 1]
        coeffs_desc = np.poly(x)
        p = coeffs_desc[::-1]
        
        # c = A_T_inv @ p
        c = A_T_inv @ p
        
        # Norm squared: sum c_k^2 * (k+1)
        norm_sq = np.sum(c**2 * laguerre_weights)
        
        if norm_sq <= 1e-10 or not np.isfinite(norm_sq):
            return 0.0
            
        return norm_sq

    def _objective(log_roots):
        """
        Minimize the upper bound on C4.
        Bound ~ 1 / ||P||^2.
        So we minimize 1/norm_sq, which is equivalent to maximizing norm_sq.
        We return 1/norm_sq.
        """
        if np.any(np.isnan(log_roots)) or np.any(np.isinf(log_roots)):
            return 1e10
        
        log_roots_clipped = np.clip(log_roots, -10, 30)
        roots = np.exp(log_roots_clipped)
        
        if np.any(roots <= 0) or np.any(roots > 1e12):
            return 1e10
            
        norm_sq = _compute_norm_sq(roots)
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    def _objective_grad(log_roots):
        """
        Objective and gradient for L-BFGS-B.
        """
        val = _objective(log_roots)
        if val >= 1e9:
            return val, np.zeros_like(log_roots)
        
        # Numerical gradient
        h = 1e-7
        grad = np.zeros_like(log_roots)
        for i in range(len(log_roots)):
            pert = np.zeros_like(log_roots)
            pert[i] = h
            grad[i] = (_objective(log_roots + pert) - val) / h
            
        return val, grad

    best_score = 1e10
    best_log_roots = None
    best_roots = None
    
    # Strategy 1: Global Search with Differential Evolution
    # Use multiple seeds and settings to explore the landscape.
    
    # Bounds for log_roots
    # Roots should be positive.
    # Based on typical scales, log range [-5, 15] covers [0.006, 3.3e6]
    bounds = [(-5.0, 15.0)] * m
    
    seeds = [42, 123, 456, 789, 101, 202, 303, 404]
    
    for seed in seeds:
        if not time_ok():
            break
        
        try:
            # Use smaller popsize and maxiter for speed, relying on multiple restarts
            result = differential_evolution(
                _objective, 
                bounds, 
                seed=seed, 
                maxiter=100, 
                popsize=20, 
                tol=1e-8, 
                polish=True,  # L-BFGS-B polish
                init='sobol' # Better initial distribution
            )
            
            if result.fun < best_score and result.fun < 1e9 and np.all(np.isfinite(result.x)):
                best_score = result.fun
                best_log_roots = result.x.copy()
                best_roots = np.sort(np.exp(np.clip(result.x, -10, 30)))
                
        except Exception:
            continue

    # Strategy 2: Local Refinement with L-BFGS-B from multiple initial points
    # This helps if DE gets stuck or if the landscape has sharp features.
    
    if time_ok():
        initial_log_roots_list = [
            np.log(np.array([(i + 1) ** 1.5 for i in range(m)])),
            np.log(np.array([(i + 1) ** 2.0 for i in range(m)])),
            np.log(np.array([(i + 1) ** 1.2 for i in range(m)])),
            np.log(np.array([3.0 + 2.0 * i for i in range(m)])),
            np.log(np.array([5.0 + 3.0 * i for i in range(m)])),
            np.log(np.array([2.0 + 8.0 * np.exp(-0.3 * i) for i in range(m)])),
            np.log(np.array([1.0 + 0.5 * i for i in range(6)] + [8.0 + 2.0 * i for i in range(6)])),
            np.log(np.array([0.5 + i * 0.8 for i in range(m)])),
            np.log(np.array([4.0 * (2.0 ** i) for i in range(m)])), # Geometric seed
        ]
        
        for x0 in initial_log_roots_list:
            if not time_ok():
                break
            try:
                res = minimize(_objective, x0, method='L-BFGS-B', jac=_objective_grad,
                              options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12,
                                       'maxls': 50})
                if res.fun < best_score and res.fun < 1e9 and np.all(np.isfinite(res.x)):
                    best_score = res.fun
                    best_log_roots = res.x.copy()
                    best_roots = np.sort(np.exp(np.clip(res.x, -10, 30)))
            except Exception:
                continue

    # Strategy 3: Nelder-Mead Polish
    if time_ok() and best_log_roots is not None:
        try:
            res = minimize(_objective, best_log_roots, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12,
                                   'adaptive': True})
            if res.fun < best_score and res.fun < 1e9 and np.all(np.isfinite(res.x)):
                best_score = res.fun
                best_log_roots = res.x.copy()
                best_roots = np.sort(np.exp(np.clip(res.x, -10, 30)))
        except Exception:
            pass

    # Strategy 4: Random Hops / Perturbations
    if time_ok() and best_log_roots is not None:
        rng = np.random.default_rng(12345)
        n_hops = 50
        for hop in range(n_hops):
            if not time_ok():
                break
            perturbation = rng.normal(0, 1.0, size=m)
            perturbed_log = best_log_roots + perturbation
            try:
                res = minimize(_objective, perturbed_log, method='L-BFGS-B', jac=_objective_grad,
                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
                if res.fun < best_score and res.fun < 1e9 and np.all(np.isfinite(res.x)):
                    best_score = res.fun
                    best_log_roots = res.x.copy()
                    best_roots = np.sort(np.exp(np.clip(res.x, -10, 30)))
            except Exception:
                continue

    # Fallback
    if best_roots is None:
        best_roots = np.array([(i + 1) ** 1.5 for i in range(m)])
        best_roots = np.sort(best_roots)
        
    best_roots = np.maximum(best_roots, 1e-6)
    best_roots = np.sort(best_roots)
    
    return list(best_roots)
# EVOLVE-BLOCK-END
