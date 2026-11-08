import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

    m = 12
    n = 1

    # 1. Precomputation: Transformation matrix from standard basis to Laguerre basis
    # We represent P(x) = prod(x - r_i) in the Laguerre basis L_k^{(1)}.
    # P(x) = sum_{k=0}^{12} c_k L_k^{(1)}(x)
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k,j] = (-1)^j * C(k+1, j+1) / j!
    
    A = np.zeros((13, 13))
    for k in range(13):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # p (coeffs in standard basis, ascending) relates to c (Laguerre coeffs) by:
    # p_j = sum_{k=j}^{12} A[k,j] c_k
    # So p = A^T c  =>  c = (A^T)^{-1} p
    
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    # Precompute weights for norm calculation
    # Orthogonality: <L_k, L_l> = delta_kl * (k + alpha + 1) for alpha=1
    # So ||P||^2 = sum_{k=0}^{12} c_k^2 * (k+1)
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
        # np.poly(roots) returns coefficients in descending order
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]  # Ascending order: p_0, p_1, ..., p_12
        
        # Compute Laguerre coefficients c
        c = A_T_inv @ p
        
        # Compute norm squared in Laguerre basis
        norm_sq = np.sum(c**2 * weights)
        
        if norm_sq <= 1e-10:
            return 1e20
        
        return 1.0 / norm_sq

    # 2. Optimization Strategy
    # We will use Differential Evolution with a larger population and more iterations,
    # followed by local polishing with L-BFGS-B.
    # The previous score was 0.9888. We aim to beat this.
    
    best_val = 1e20
    best_log_roots = None

    # Bounds for log roots
    # Roots must be positive.
    # Based on the seed [4, 8, 16, ...], log range is roughly [1.4, 9.2].
    # We will explore a wider range: [0, 12] for log(roots).
    bounds_log = [(0.0, 12.0)] * 12

    def update_best(val, x):
        nonlocal best_val, best_log_roots
        if val < best_val:
            best_val = val
            best_log_roots = x.copy()

    # --- Phase 1: Differential Evolution ---
    # Run DE with a robust configuration.
    # We use a larger population and more iterations to ensure good exploration.
    # We also use 'sobol' initialization for better space coverage.
    
    de_configs = [
        {'seed': 1, 'maxiter': 200, 'popsize': 20, 'tol': 1e-8, 'init': 'sobol'},
        {'seed': 2, 'maxiter': 200, 'popsize': 20, 'tol': 1e-8, 'init': 'latinhypercube'},
        {'seed': 3, 'maxiter': 150, 'popsize': 25, 'tol': 1e-8, 'init': 'sobol'},
    ]

    for config in de_configs:
        if time.time() - start_time > deadline - 30:
            break
        try:
            res = differential_evolution(
                compute_cost, 
                bounds_log, 
                polish=False, # We polish separately
                **config
            )
            # Polish the result
            if time.time() - start_time < deadline - 10:
                try:
                    res_polish = minimize(
                        compute_cost, 
                        res.x, 
                        method='L-BFGS-B', 
                        bounds=bounds_log, 
                        options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-8}
                    )
                    update_best(res_polish.fun, res_polish.x)
                except Exception:
                    update_best(res.fun, res.x)
            else:
                update_best(res.fun, res.x)
        except Exception:
            pass

    # --- Phase 2: Local Search from Diverse Initial Points ---
    # If DE didn't find a great solution, or to refine, try local search from specific points.
    
    initial_points = []
    
    # Seed 1: Geometric (original seed)
    r_seed = 4.0 * (2.0 ** np.arange(12))
    initial_points.append(np.log(r_seed))
    
    # Seed 2: Uniform in [0.1, 100]
    r_uniform = np.linspace(0.1, 100.0, 12)
    initial_points.append(np.log(r_uniform))
    
    # Seed 3: Uniform in [1, 1000]
    r_uniform2 = np.linspace(1.0, 1000.0, 12)
    initial_points.append(np.log(r_uniform2))
    
    # Random points
    rng = np.random.RandomState(42)
    for _ in range(10):
        log_r = rng.uniform(np.log(0.01), np.log(10000.0), 12)
        initial_points.append(log_r)

    for i, init_x in enumerate(initial_points):
        if time.time() - start_time > deadline - 10:
            break
        
        try:
            res_lf = minimize(
                compute_cost, 
                init_x, 
                method='L-BFGS-B', 
                bounds=bounds_log, 
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-8}
            )
            update_best(res_lf.fun, res_lf.x)
        except Exception:
            pass

    # --- Final Fallback ---
    if best_log_roots is None:
        best_log_roots = np.log(4.0 * (2.0 ** np.arange(12)))
        best_val = compute_cost(best_log_roots)

    # Convert back to roots
    best_roots = np.exp(best_log_roots)
    
    # Ensure positivity and sorting
    best_roots = np.sort(best_roots)
    best_roots = np.maximum(best_roots, 1e-10)
    
    return list(best_roots)

# Main entry point
if __name__ == "__main__":
    roots = construct()
    print(roots)
