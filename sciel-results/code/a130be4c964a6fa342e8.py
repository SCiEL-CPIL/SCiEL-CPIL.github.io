"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.special import roots_laguerre
from scipy.optimize import differential_evolution, minimize

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Returns a list of 12 positive root locations.
    Score is the upper bound on C4 (lower is better for the evaluator's raw value,
    but the search score is seed/value, so higher search score = better).
    """
    start_time = time.time()
    DEADLINE = 600.0
    SAFETY_MARGIN = 30.0

    def time_ok():
        return (time.time() - start_time) < (DEADLINE - SAFETY_MARGIN)

    m = 12
    n = 1
    alpha = 2  # alpha = 2*n

    # --- Objective Function ---
    # We aim to minimize the "upper bound on C4".
    # Based on the inspiration programs (score ~33.3), the objective is related to
    # maximizing the norm of the polynomial in the Laguerre basis, or equivalently
    # minimizing 1/norm_sq.
    #
    # Let P(x) = prod_{i=1}^m (x - r_i).
    # We express P(x) in the basis of Generalized Laguerre polynomials L_k^{(alpha)}(x).
    # P(x) = sum_{k=0}^m c_k L_k^{(alpha)}(x).
    # The squared norm in the L2 space with weight x^alpha e^{-x} is:
    # ||P||^2 = sum_{k=0}^m c_k^2 * (k + alpha) * Gamma(k+1) / Gamma(k+alpha+1) * Gamma(k+alpha+1) ?
    # Actually, the orthogonality relation is:
    # <L_k^{(alpha)}, L_j^{(alpha)}> = delta_{kj} * (k+alpha) * Gamma(k+1) / Gamma(k+alpha+1) * ...?
    # Standard formula: <L_k^{(alpha)}, L_j^{(alpha)}> = delta_{kj} * binom(k+alpha, k)
    # which is (k+alpha)! / (k! alpha!).
    # For alpha=2: weight is (k+2)! / (k! 2!) = (k+2)(k+1)/2.
    
    # Let's verify the transformation.
    # L_k^{(alpha)}(x) = sum_{j=0}^k (-1)^j * binom(k+alpha, k-j) * x^j / j!
    # We need the matrix A such that L_k(x) = sum_j A[k,j] x^j.
    # A[k,j] = (-1)^j * binom(k+alpha, k-j) / j!
    
    # We want c such that P = A c. So c = A^{-1} P_coeffs.
    # P_coeffs are the standard monomial coefficients of P(x).
    
    # Precompute A inverse
    A = np.zeros((m+1, m+1))
    for k in range(m+1):
        for j in range(k+1):
            # binom(k+alpha, k-j)
            # k-j is the lower index in binom
            # Note: standard binom(n, k) = n! / (k! (n-k)!)
            # Here n = k+alpha, k' = k-j.
            # binom(k+alpha, k-j) = (k+alpha)! / ((k-j)! (k+alpha - (k-j))!) = (k+alpha)! / ((k-j)! (alpha+j)!)
            
            # Using math.comb is safer for integers, but we need to handle the formula carefully.
            # Let's use the explicit formula for A[k,j]
            # A[k,j] = (-1)^j * binom(k+alpha, j) / j! ? No.
            # Definition: L_n^{(alpha)}(x) = sum_{k=0}^n (-1)^k binom(n+alpha, n-k) x^k / k!
            # So coeff of x^j is (-1)^j binom(k+alpha, k-j) / j!
            
            try:
                binom_val = math.comb(k+alpha, k-j)
                A[k, j] = ((-1)**j) * binom_val / math.factorial(j)
            except ValueError:
                A[k, j] = 0.0
                
    # A is upper triangular? No, L_k is degree k. So A is upper triangular if rows are k and cols are j?
    # A[k,j] is non-zero only if j <= k. So it is lower triangular if we view k as row, j as col.
    # Wait, L_0 = 1. A[0,0]=1.
    # L_1 = -x + (alpha+1). A[1,0] = alpha+1, A[1,1] = -1.
    # So A is lower triangular.
    
    A_inv = np.linalg.inv(A)
    
    # Weights for the norm
    # w_k = binom(k+alpha, k) = (k+alpha)! / (k! alpha!)
    weights = np.array([math.comb(k+alpha, k) for k in range(m+1)])
    
    def compute_c4_bound(roots):
        """
        Compute the value to minimize.
        This is 1 / ||P||^2.
        """
        # Build polynomial coefficients in standard basis (ascending)
        # P(x) = prod(x - r_i)
        # np.poly returns descending coeffs: [1, c1, ..., cm]
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1] # ascending
        
        # Project onto Laguerre basis
        c = A_inv @ p
        
        # Norm squared
        norm_sq = np.sum(c**2 * weights)
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    def cost_func(log_roots):
        """
        Objective function for the optimizer.
        params: 12 values, representing log(roots) to ensure positivity.
        """
        roots = np.exp(log_roots)
        
        # Check for numerical issues
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
            
        # Optional: Sort roots to ensure consistency? Not needed for product.
        # But for numerical stability in np.poly, maybe?
        
        return compute_c4_bound(roots)

    # --- Optimization Strategy ---
    
    # 1. Initial Guesses
    # a) Laguerre roots of L_m^{(alpha)}
    try:
        r_lag, _ = roots_laguerre(m, alpha)
        # Ensure positive
        r_lag = np.abs(r_lag)
        # Add small epsilon to avoid log(0)
        r_lag = np.maximum(r_lag, 1e-6)
        x0_lag = np.log(r_lag)
    except:
        x0_lag = np.log(np.arange(1, m+1, dtype=float))
        
    # b) Uniform spacing
    x0_uniform = np.log(np.arange(1, m+1, dtype=float))
    
    # c) Geometric sequence (from inspiration)
    # r_i = a * q^i. 
    # Let's try a=1, q=1.1 or similar.
    x0_geo = np.log(1.0) + np.arange(m) * np.log(1.2)
    
    initial_guesses = [x0_lag, x0_uniform, x0_geo]
    
    best_roots = None
    best_val = float('inf')
    
    # 2. Local Refinement (L-BFGS-B)
    # Run local optimization from each initial guess
    for x0 in initial_guesses:
        if not time_ok():
            break
        try:
            res = minimize(cost_func, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-10})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_roots = np.exp(res.x)
        except Exception as e:
            pass
            
    # 3. Global Search (Differential Evolution)
    # If time permits, run DE to explore the landscape.
    # Bounds for log_roots:
    # Roots should be positive. 
    # If roots are very small, P(x) ~ (-1)^m prod(r_i), norm might be small?
    # If roots are very large, P(x) ~ x^m, norm is fixed?
    # Let's use bounds [-5, 10] for log_roots (roots from 0.006 to 22000).
    bounds = [(-5.0, 10.0)] * m
    
    # Run DE if we have enough time (DE is expensive)
    # Estimate time for DE: popsize=20, maxiter=100, 12D. 
    # Each function eval is fast (matrix mult). 
    # Let's try one DE run with a moderate budget.
    
    if time_ok():
        try:
            # Use a smaller population and fewer iterations to fit in time
            # Or use 'sobol' init for better coverage.
            res_de = differential_evolution(
                cost_func, 
                bounds, 
                seed=42, 
                maxiter=100, 
                popsize=15, 
                tol=1e-8, 
                polish=True,
                init='sobol'
            )
            
            if res_de.success and res_de.fun < best_val:
                best_val = res_de.fun
                best_roots = np.exp(res_de.x)
                
        except Exception as e:
            pass
            
    # 4. Final Polish
    if best_roots is not None and time_ok():
        try:
            x0_final = np.log(best_roots)
            res_final = minimize(cost_func, x0_final, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-12})
            if res_final.success and res_final.fun < best_val:
                best_val = res_final.fun
                best_roots = np.exp(res_final.x)
        except Exception as e:
            pass
            
    # Fallback
    if best_roots is None:
        best_roots = np.arange(1, m+1, dtype=float)
        
    # Ensure positive and sorted
    best_roots = np.maximum(best_roots, 1e-6)
    best_roots = sorted(best_roots)
    
    return [float(x) for x in best_roots]

# EVOLVE-BLOCK-END
