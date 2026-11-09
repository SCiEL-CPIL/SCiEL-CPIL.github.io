import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_laguerre
import time
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 590

    # Task specifies 12 double roots, implying degree 24 polynomial
    m = 12
    N = 2 * m  # Degree of polynomial

    # Precomputation: Transformation matrix from standard basis to Laguerre basis
    # P(x) = sum_{k=0}^{N} c_k L_k^{(1)}(x)
    # We need to express P(x) = prod (x-r_i)^2 = sum_{j=0}^{N} p_j x^j
    # in terms of Laguerre coefficients.
    # L_k^{(1)}(x) = sum_{j=0}^k a_{k,j} x^j
    # c = A^{-1} p, where A[k,j] = a_{k,j}
    
    A = np.zeros((N + 1, N + 1))
    for k in range(N + 1):
        for j in range(k + 1):
            # Coefficients of Laguerre polynomial L_k^{(1)}(x)
            # Formula: L_k^{(\alpha)}(x) = sum_{j=0}^k (-1)^j binom(k+alpha, k-j) / j! x^j
            # Here alpha = 1
            # Term j: (-1)^j * binom(k+1, k-j) / j!
            # Note: binom(k+1, k-j) = binom(k+1, j+1)
            A[k, j] = ((-1)**j) * math.comb(k + 1, j + 1) / math.factorial(j)
            
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        A_inv = np.linalg.pinv(A)

    weights = np.arange(N + 1) + 1  # w_k = k+1 for L_k^{(1)}? 
    # Wait, the weight for L_k^{(alpha)} in the inner product <f,g> = int_0^inf f g x^alpha e^{-x} dx is 1.
    # The norm squared of L_k^{(alpha)} is binom(k+alpha, k).
    # For alpha=1, norm^2 = k+1.
    # So if P = sum c_k L_k, then ||P||^2 = sum c_k^2 (k+1).
    # Yes, weights = np.arange(N+1) + 1 is correct.

    def compute_cost_from_roots(roots):
        """Compute cost given positive roots."""
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e20
        
        # Construct polynomial coefficients p_j for P(x) = prod (x-r_i)^2
        # Start with 1, multiply by (x-r_i)^2 for each root
        p = np.array([1.0])
        for r in roots:
            # (x-r)^2 = x^2 - 2rx + r^2
            # Convolve p with [r^2, -2r, 1]
            # Note: np.poly returns descending powers. Let's stick to ascending for consistency with A.
            # p is ascending: p[0] + p[1]x + ...
            # Factor is r^2 - 2r x + x^2 -> coeff [r^2, -2r, 1]
            new_p = np.convolve(p, [r**2, -2*r, 1])
            p = new_p
        
        if len(p) != N + 1:
             # Should always be N+1 if we start with 1 and multiply by 12 quadratic terms
             return 1e20

        # Convert to Laguerre coefficients
        c = A_inv @ p
        
        norm_sq = np.sum(c**2 * weights)
        
        if norm_sq <= 1e-10:
            return 1e20
        
        return 1.0 / norm_sq

    def compute_cost(log_roots):
        """
        Compute cost given log roots.
        """
        if time.time() > deadline:
            return 1e20
        try:
            roots = np.exp(log_roots)
            return compute_cost_from_roots(roots)
        except Exception:
            return 1e20

    # --- Optimization Strategy ---
    # 1. Global Search with Differential Evolution
    # 2. Local Polish with L-BFGS-B
    
    # Initial bounds for log roots
    # Roots are typically in [0, 2*m+1] or similar for Laguerre.
    # Let's bound log roots to [log(0.01), log(100)]
    log_bounds = [(math.log(0.01), math.log(100.0))] * m

    # Seed for reproducibility
    rng_seed = 42
    
    # Run Differential Evolution
    # maxiter=100, popsize=20 is a good balance for 12 dimensions
    # The deadline is 590s. DE can be slow, so we monitor time.
    
    best_val = 1e20
    best_log_roots = None
    
    def update_best(val, x):
        nonlocal best_val, best_log_roots
        if val < best_val:
            best_val = val
            best_log_roots = x.copy()

    # Helper to check time and return best
    def time_check():
        return time.time() > deadline - 10

    # Use a wrapper for DE to stop early if needed?
    # scipy.optimize.differential_evolution doesn't have a great callback for stopping based on time inside the loop easily without wrapping.
    # We can use a smaller popsize and maxiter to keep it fast, then polish.
    
    try:
        # DE bounds
        de_bounds = log_bounds
        
        # Run DE
        # tol=1e-8, maxiter=50
        res_de = differential_evolution(
            compute_cost, 
            de_bounds, 
            seed=rng_seed,
            maxiter=50,
            popsize=15,
            tol=1e-8,
            polish=True,  # DE does a polish at the end
            init='sobol'
        )
        
        if res_de.fun < best_val:
            update_best(res_de.fun, res_de.x)
            
    except Exception as e:
        print(f"DE failed: {e}")
        pass

    if time_check():
        if best_log_roots is not None:
            roots = np.exp(best_log_roots)
            return list(roots)
        else:
            # Fallback to Laguerre roots of degree 12 (scaled?)
            # If we must return something valid
            l_roots, _ = roots_laguerre(12, 1)
            return list(l_roots)

    # --- Phase 2: Multi-start Local Optimization ---
    # Generate diverse starting points for L-BFGS-B
    # 1. Result from DE
    # 2. Random points in log space
    # 3. Points based on Laguerre roots of degree 12 (mapped to 12 roots?)
    #    Note: Laguerre roots of degree 12 are 12 roots. We have 12 double roots.
    #    So we can use the 12 Laguerre roots of degree 12 as initial guesses for the 12 double roots.
    
    start_points = []
    
    if best_log_roots is not None:
        start_points.append(best_log_roots)
    
    # Add Laguerre roots of degree 12
    try:
        l_roots, _ = roots_laguerre(12, 1)
        l_roots = np.sort(l_roots)
        log_l_roots = np.log(l_roots)
        start_points.append(log_l_roots)
    except Exception:
        pass

    # Add some random starts
    rng = np.random.RandomState(rng_seed)
    for _ in range(5):
        random_log_roots = rng.uniform(math.log(0.1), math.log(50.0), size=m)
        start_points.append(random_log_roots)
    
    # Sort start points by cost to prioritize good ones
    # (Optional, but helps if we run out of time)
    # start_points.sort(key=lambda x: compute_cost(x))
    
    local_opts = {'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-10}
    
    for x0 in start_points:
        if time.time() > deadline - 5:
            break
        
        try:
            res_loc = minimize(
                compute_cost, 
                x0, 
                method='L-BFGS-B', 
                options=local_opts
            )
            
            if res_loc.fun < best_val:
                update_best(res_loc.fun, res_loc.x)
                
        except Exception:
            pass

    # --- Final Polish ---
    if best_log_roots is not None:
        try:
            res_final = minimize(
                compute_cost, 
                best_log_roots, 
                method='L-BFGS-B', 
                options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            if res_final.fun < best_val:
                best_val = res_final.fun
                best_log_roots = res_final.x
        except Exception:
            pass

    if best_log_roots is not None:
        best_roots = np.exp(best_log_roots)
        # Ensure positive
        best_roots = np.maximum(best_roots, 1e-10)
        return list(best_roots)
    else:
        # Fallback
        l_roots, _ = roots_laguerre(12, 1)
        return list(l_roots)

if __name__ == "__main__":
    roots = construct()
    print(roots)
