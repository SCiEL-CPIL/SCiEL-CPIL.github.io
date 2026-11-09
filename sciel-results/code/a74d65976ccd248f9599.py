import numpy as np
from scipy.optimize import differential_evolution, minimize, basinhopping
import time
import math

def construct():
    start_time = time.time()
    # Hard deadline: leave margin for the 630s kill limit.
    # We target finishing within 580s to be safe.
    deadline = start_time + 580

    m = 12
    n = 1

    # Precomputation: Transformation matrix from standard basis to Laguerre basis
    # We want to represent P(x) = prod(x - r_i) in the Laguerre basis L_k^{(1)}.
    # P(x) = sum_{k=0}^{12} c_k L_k^{(1)}(x)
    # We need to compute c from the coefficients of P(x) in standard basis.
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k,j] = (-1)^j * C(k+1, j+1) / j!
    
    A = np.zeros((13, 13))
    for k in range(13):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # p (coeffs in standard basis, ascending) and c (coeffs in Laguerre basis).
    # p = A^T c
    # So c = (A^T)^{-1} p.
    
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    # Weights for the Laguerre norm: <L_k, L_l> = delta_kl * (k+1) for alpha=1
    weights = np.arange(13) + 1

    def compute_cost(log_roots):
        """
        Compute the cost function (upper bound on C4) given log roots.
        Cost = 1 / ||P||^2_Laguerre
        We want to minimize this, which means maximizing ||P||^2.
        """
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return 1e20

        roots = np.exp(log_roots)
        
        # Check validity
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e20

        # Build polynomial P(x) = prod(x - r_i)
        # np.poly(roots) returns coefficients in descending order: [1, -sum(r), sum(r_i r_j), ...]
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]  # Ascending order: p_0, p_1, ..., p_12
        
        # Compute Laguerre coefficients c
        c = A_T_inv @ p
        
        # Compute norm squared in Laguerre basis
        norm_sq = np.sum(c**2 * weights)
        
        if norm_sq <= 1e-10:
            return 1e20
        
        return 1.0 / norm_sq

    # Optimization Strategy:
    # 1. Single large Differential Evolution run (better exploration than many small ones).
    # 2. Basin Hopping from the best DE result and the seed.
    # 3. Final high-precision polish.

    best_val = 1e20
    best_log_roots = None

    # Define bounds for log roots
    # Roots must be positive.
    # Seed: 4 * 2^i -> log(4) + i*log(2). Range ~ [1.4, 9.2].
    # Allow wider range to find better basins: [-5, 15]
    bounds_log = [(-5.0, 15.0)] * 12

    # Helper to update best
    def update_best(val, x):
        nonlocal best_val, best_log_roots
        if val < best_val:
            best_val = val
            best_log_roots = x.copy()

    # --- Phase 1: Global Search (Single Large DE) ---
    # Based on findings, a single longer DE with larger population is more effective.
    if time.time() - start_time < deadline - 100:
        try:
            # Use a large population and sufficient iterations
            res = differential_evolution(
                compute_cost, 
                bounds_log, 
                seed=42, 
                maxiter=200, 
                popsize=30, 
                tol=1e-10, 
                polish=True, # Built-in L-BFGS-B polish
                init='sobol'
            )
            update_best(res.fun, res.x)
        except Exception as e:
            pass

    # --- Phase 2: Basin Hopping ---
    # Start from the best DE result and the original seed
    initial_points = []
    
    # Seed 1: Geometric (Original Seed)
    r_seed = 4.0 * (2.0 ** np.arange(12))
    initial_points.append(np.log(r_seed))
    
    # Add best from DE if available
    if best_log_roots is not None:
        initial_points.append(best_log_roots.copy())
        
    # Add a few random points
    rng = np.random.RandomState(123)
    for _ in range(5):
        log_r = rng.uniform(-2.0, 12.0, 12)
        initial_points.append(log_r)

    for i, init_x in enumerate(initial_points):
        if time.time() - start_time > deadline - 50:
            break
        
        try:
            # Define step size for basin hopping
            step_size = 1.0
            
            def objective(x):
                return compute_cost(x)
                
            # Basin hopping can be slow, so limit iterations
            res_bh = basinhopping(
                objective, 
                init_x, 
                step_size=step_size, 
                minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': bounds_log},
                niter=20, # Limit iterations to keep time down
                seed=100+i,
                accept_test=lambda new, old: new < old # Only accept improvements
            )
            update_best(res_bh.fun, res_bh.x)
        except Exception as e:
            pass

    # --- Phase 3: Final Polish ---
    # Polish the global best found so far
    if best_log_roots is not None and time.time() - start_time < deadline - 10:
        try:
            # L-BFGS-B for smooth convergence
            res_lf = minimize(
                compute_cost, 
                best_log_roots, 
                method='L-BFGS-B', 
                bounds=bounds_log, 
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            update_best(res_lf.fun, res_lf.x)
            
            # Nelder-Mead for fine-tuning if L-BFGS-B got stuck
            if time.time() - start_time < deadline - 5:
                res_nm = minimize(
                    compute_cost, 
                    res_lf.x, 
                    method='Nelder-Mead', 
                    options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12}
                )
                update_best(res_nm.fun, res_nm.x)
        except Exception as e:
            pass

    # --- Final Fallback ---
    if best_log_roots is None:
        # If everything failed, return the original seed
        best_log_roots = np.log(4.0 * (2.0 ** np.arange(12)))
        best_val = compute_cost(best_log_roots)

    # Convert back to roots
    best_roots = np.exp(best_log_roots)
    
    # Ensure positivity and sorting (though order doesn't matter for product)
    best_roots = np.sort(best_roots)
    best_roots = np.maximum(best_roots, 1e-10)
    
    return list(best_roots)

# Main entry point
if __name__ == "__main__":
    # For local testing
    roots = construct()
    print(roots)
