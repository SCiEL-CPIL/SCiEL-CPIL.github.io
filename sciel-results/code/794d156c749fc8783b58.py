import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_genlaguerre
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 570  # Safety margin before 600s limit

    m = 12
    n = 1
    alpha = 1

    # Precompute A matrix: A[k,j] = coeff of x^j in L_k^{(1)}(x)
    # A[k,j] = (-1)^j * binom(k+1, j+1) / j!
    A = np.zeros((m + 1, m + 1))
    for k in range(m + 1):
        for j in range(k + 1):
            A[k, j] = ((-1.0)**j) * math.comb(k + 1, j + 1) / math.factorial(j)
    
    # A is upper triangular. A^T is lower triangular.
    # We need to solve A^T c = p, where p is the monomial coefficients (ascending)
    # c = (A^T)^{-1} p
    # Precompute the inverse of A^T for speed
    try:
        AT_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        AT_inv = np.linalg.pinv(A.T)

    def compute_norm_sq(roots):
        """
        Compute the squared norm of the polynomial in the Laguerre basis.
        roots: array of 12 positive numbers.
        """
        # Build polynomial coefficients in standard basis (ascending)
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1].copy()  # ascending order
        
        # c = AT_inv @ p
        c = AT_inv @ p
        
        # Norm squared: ||P||^2 = sum c_k^2 * (k+1)
        norm_sq = np.sum(c**2 * (np.arange(m + 1) + 1))
        
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_func(log_roots):
        """
        Minimize the upper bound on C4 = 1/||P||^2.
        """
        roots = np.exp(log_roots)
        
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
            
        norm_sq = compute_norm_sq(roots)
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    # Seed: geometric sequence a=4, q=2
    seed_log_roots = np.log(4.0) + np.arange(12) * np.log(2.0)
    seed_roots = np.exp(seed_log_roots)
    seed_val = cost_func(seed_log_roots)
    
    # Bounds for log_roots
    bounds = [(0.5, 12.0)] * 12
    
    best_roots = seed_roots.copy()
    best_val = seed_val
    best_log_roots = seed_log_roots.copy()

    # Run Differential Evolution with multiple restarts
    seeds = [42, 123, 456, 789, 101, 202]
    
    for seed in seeds:
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break
        
        try:
            result = differential_evolution(
                cost_func, 
                bounds, 
                seed=seed, 
                maxiter=150, 
                popsize=15, 
                tol=1e-8, 
                polish=True,
                init='sobol',
                updating='deferred'
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x.copy()
                best_roots = np.exp(best_log_roots)
                
        except Exception:
            pass

    # Final high-precision polish pass
    if time.time() - start_time < deadline - 5 and best_log_roots is not None:
        try:
            polish_result = minimize(
                cost_func, best_log_roots, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_log_roots = polish_result.x.copy()
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    # Ensure roots are positive
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    return [4.0 * (2.0 ** i) for i in range(12)]
