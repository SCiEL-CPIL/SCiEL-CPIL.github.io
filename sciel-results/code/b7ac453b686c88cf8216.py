import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math

def construct():
    start_time = time.time()
    # The evaluator kills after 630s. We set a hard deadline at 600s to ensure
    # we have time to return the result and flush any output.
    deadline = start_time + 600

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
    # A is lower triangular, so A.T is upper triangular.
    # We can use np.linalg.inv or solve, but since we do this once, inv is fine.
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
    # 1. Global search using Differential Evolution (DE).
    # 2. Polish the best result with L-BFGS-B.
    
    # Define bounds for log_roots
    # Roots in [1e-3, 1e4] -> logs in [-6.9, 9.2]
    # Let's use a slightly tighter but safe range to help DE converge faster.
    # Based on the seed 4*2^i, logs are ~1.4 to ~9.
    # Let's allow [0.0, 10.0] for logs, i.e., roots [1, 22026].
    # Actually, let's be a bit wider to be safe: [ -2.0, 12.0 ]
    log_low = -2.0
    log_high = 12.0
    bounds = [(log_low, log_high)] * 12

    best_val = 1e10
    best_log_roots = None
    best_roots = None

    # Initial guess: The geometric sequence from the seed
    seed_log_roots = np.log(4.0) + np.arange(12) * np.log(2.0)
    
    # Ensure seed is within bounds
    seed_log_roots = np.clip(seed_log_roots, log_low, log_high)
    
    # Run Differential Evolution
    # We allocate most of the time to DE.
    # popsize=15, maxiter=500 is a good start. 
    # The callback will check time and update best_val.
    
    def de_callback(xk, convergence):
        nonlocal best_val, best_log_roots, best_roots
        val = cost_func(xk)
        if val < best_val:
            best_val = val
            best_log_roots = xk.copy()
            best_roots = np.exp(xk)
        
        if time.time() > deadline - 30:
            raise RuntimeError("Time limit reached")

    try:
        # Using 'sobol' initialization for better coverage
        de_result = differential_evolution(
            cost_func,
            bounds,
            seed=42,
            maxiter=500,
            popsize=15,
            tol=1e-8,
            polish=True, # Let DE do a final polish as well
            init='sobol',
            callback=de_callback,
            update_interval=1 # Update callback every iteration
        )
        
        # Update best from DE result
        if de_result.fun < best_val:
            best_val = de_result.fun
            best_log_roots = de_result.x
            best_roots = np.exp(de_result.x)
            
    except RuntimeError:
        pass
    except Exception as e:
        # If DE fails completely, fallback to seed
        if best_log_roots is None:
            best_log_roots = seed_log_roots
            best_val = cost_func(seed_log_roots)
            best_roots = np.exp(seed_log_roots)

    # Final Polish with L-BFGS-B if we have time
    if time.time() < deadline - 10 and best_log_roots is not None:
        try:
            polish_result = minimize(
                cost_func, 
                best_log_roots, 
                method='L-BFGS-B',
                bounds=bounds, 
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_log_roots = polish_result.x
                best_roots = np.exp(polish_result.x)
        except Exception:
            pass

    # If we still have time and haven't found a good solution, try a few random restarts with L-BFGS-B
    # This is a fallback if DE didn't run long enough or got stuck.
    if time.time() < deadline - 5:
        rng = np.random.default_rng(123)
        for _ in range(5):
            if time.time() > deadline - 2:
                break
            init = rng.uniform(log_low, log_high, size=12)
            try:
                res = minimize(
                    cost_func, init, method='L-BFGS-B',
                    bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_log_roots = res.x
                    best_roots = np.exp(res.x)
            except Exception:
                pass

    # Ensure roots are positive and sorted (order doesn't matter for product, but good for consistency)
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(12)]
