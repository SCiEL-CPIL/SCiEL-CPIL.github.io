import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_genlaguerre
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin before 600s limit

    m = 12
    n = 1

    # Precompute the transformation matrix from standard basis to Laguerre basis
    # P(x) = sum c_k L_k^{(1)}(x)
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    
    A = np.zeros((13, 13))  # A[k, j] = coeff of x^j in L_k^{(1)}(x)
    for k in range(13):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # Solve for c given p. c = (A^T)^{-1} p
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    def compute_norm_sq(roots):
        """
        Compute the squared norm of the polynomial in the Laguerre basis.
        roots: array of 12 positive numbers.
        """
        # Build polynomial coefficients in standard basis (ascending)
        # P(x) = prod(x - r_i)
        # np.poly returns descending coeffs: [1, c1, ..., cm]
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1] # ascending
        
        # c = A_T_inv @ p
        c = A_T_inv @ p
        
        # Norm squared: <L_k, L_l> = delta_kl * (k+1) for alpha=1
        # ||P||^2 = sum c_k^2 * (k+1)
        norm_sq = np.sum(c**2 * (np.arange(13) + 1))
        
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
    # Multi-start L-BFGS-B with diverse initial guesses.
    # This is much faster than DE for 12D smooth problems.
    
    best_roots = None
    best_val = 1e10
    best_log_roots = None

    # Define bounds for log_roots
    # Roots in [0.1, 1000] -> logs in [-2.3, 6.9]
    # Let's widen slightly to be safe: [1e-3, 1e4] -> [-6.9, 9.2]
    log_low = -6.0
    log_high = 9.0
    bounds_log = (log_low, log_high)
    bounds = [(bounds_log, bounds_log)] * 12

    # Generate diverse initial guesses
    rng = np.random.default_rng(42)
    
    # 1. The original seed
    initial_guesses = []
    seed_log_roots = np.log(4.0) + np.arange(12) * np.log(2.0)
    initial_guesses.append(seed_log_roots)
    
    # 2. Random log-uniform starts
    for _ in range(100):
        # Sample logs uniformly in a reasonable range
        # Center around log(10) ~ 2.3, width +/- 4
        # Ensure we stay within bounds
        low_sample = max(log_low, log_low - 2)
        high_sample = min(log_high, log_high + 2)
        logs = rng.uniform(low_sample, high_sample, size=12)
        # Clip to bounds
        logs = np.clip(logs, log_low, log_high)
        initial_guesses.append(logs)
        
    # 3. Some structured guesses (geometric with different ratios)
    for base in [2.0, 3.0, 5.0, 0.5, 1.5]:
        for start in [1.0, 5.0, 10.0]:
            logs = np.log(start) + np.arange(12) * np.log(base)
            if np.all(logs > log_low) and np.all(logs < log_high):
                initial_guesses.append(logs)

    # Run L-BFGS-B on each guess
    for i, init in enumerate(initial_guesses):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
            
        try:
            result = minimize(
                cost_func, init, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x
                best_roots = np.exp(best_log_roots)
                
        except Exception as e:
            pass

    # If we still have time, try a few DE runs with a tight budget to see if we can escape local minima
    if time.time() - start_time < deadline - 30:
        try:
            # Use a small population and few iterations
            result_de = differential_evolution(
                cost_func, 
                bounds, 
                seed=123, 
                maxiter=50, 
                popsize=10, 
                tol=1e-8, 
                polish=True,
                init='sobol'
            )
            if result_de.fun < best_val:
                best_val = result_de.fun
                best_log_roots = result_de.x
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    # Final polish
    if time.time() - start_time < deadline - 5 and best_log_roots is not None:
        try:
            polish_result = minimize(
                cost_func, best_log_roots, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_log_roots = polish_result.x
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    # Ensure roots are positive and sorted (though order doesn't matter for the product)
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(12)]
