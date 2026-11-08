import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_genlaguerre
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 590

    m = 12
    n = 1

    # Precompute the transformation matrix from standard basis to Laguerre basis
    # P(x) = sum c_k L_k^{(1)}(x)
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    
    A = np.zeros((13, 13))
    for k in range(13):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    def compute_norm_sq_from_std_coeffs(p_std):
        """
        p_std: array of 13 coefficients in ascending order [c0, c1, ..., c12]
        """
        c = A_T_inv @ p_std
        norm_sq = np.sum(c**2 * (np.arange(13) + 1))
        return norm_sq

    def cost_func_6_double(log_q):
        """
        Optimize 6 roots q_i such that P(x) = prod(x-q_i)^2.
        """
        q = np.exp(log_q)
        if np.any(q <= 0) or np.any(np.isnan(q)) or np.any(np.isinf(q)):
            return 1e10

        # Q(x) = prod(x - q_i)
        q_coeffs_desc = np.poly(q)
        q_std = q_coeffs_desc[::-1]
        
        # P(x) = Q(x)^2
        # Convolve q_std with itself
        p_std = np.convolve(q_std, q_std)
        
        # p_std should have length 13 (degree 12)
        if len(p_std) != 13:
            return 1e10
            
        norm_sq = compute_norm_sq_from_std_coeffs(p_std)
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    def cost_func_12_simple(log_r):
        """
        Fallback: Optimize 12 roots r_i such that P(x) = prod(x-r_i).
        """
        r = np.exp(log_r)
        if np.any(r <= 0) or np.any(np.isnan(r)) or np.any(np.isinf(r)):
            return 1e10

        coeffs_desc = np.poly(r)
        p_std = coeffs_desc[::-1]
        
        if len(p_std) != 13:
            return 1e10

        norm_sq = compute_norm_sq_from_std_coeffs(p_std)
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    # Strategy:
    # 1. Try the 6-double-root hypothesis (P = Q^2, deg Q = 6).
    # 2. If it fails or is worse than the 12-simple-root baseline, use 12-simple.
    
    best_val = 1e10
    best_roots = None
    best_type = None

    # --- Attempt 1: 6 Double Roots ---
    # Seed: geometric sequence for q
    seed_log_q = np.log(4.0) + np.arange(6) * np.log(2.0)
    bounds_q = [(-4.0, 10.0)] * 6

    # Run DE for 6 roots
    seeds = [42, 123, 456]
    for seed in seeds:
        elapsed = time.time() - start_time
        if elapsed > deadline - 120:
            break
        try:
            result = differential_evolution(
                cost_func_6_double, 
                bounds_q, 
                seed=seed, 
                maxiter=150, 
                popsize=20, 
                tol=1e-8, 
                polish=True,
                init='sobol'
            )
            if result.fun < best_val:
                best_val = result.fun
                q_roots = np.exp(result.x)
                # Convert 6 double roots to 12 locations?
                # The task asks for "list of 12 positive double-root locations".
                # If P has 6 double roots q1..q6, the "locations" might be q1..q6 repeated?
                # Or just the 6 unique locations?
                # "list of 12 ... locations".
                # If I return 6 locations, it might be invalid format.
                # If I return [q1, q1, ..., q6, q6], that's 12 items.
                # Let's return the 12 items.
                best_roots = np.concatenate([q_roots, q_roots])
                best_type = 'double'
        except Exception:
            pass

    # --- Attempt 2: 12 Simple Roots (Baseline from current program) ---
    # We need to beat the current program's score.
    # The current program used 12 simple roots.
    # Let's run a quick DE for 12 simple roots to ensure we have a strong baseline.
    
    seed_log_r = np.log(4.0) + np.arange(12) * np.log(2.0)
    bounds_r = [(-4.0, 10.0)] * 12

    seeds_r = [42, 123]
    for seed in seeds_r:
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break
        try:
            result = differential_evolution(
                cost_func_12_simple, 
                bounds_r, 
                seed=seed, 
                maxiter=100, 
                popsize=15, 
                tol=1e-8, 
                polish=True,
                init='sobol'
            )
            if result.fun < best_val:
                best_val = result.fun
                best_roots = np.exp(result.x)
                best_type = 'simple'
        except Exception:
            pass

    # Fallback to seed values if optimization failed
    if best_roots is None:
        best_roots = 4.0 * (2.0 ** np.arange(12))
        best_type = 'simple'

    # Ensure roots are positive
    best_roots = np.maximum(best_roots, 1e-6)
    
    return list(best_roots)
