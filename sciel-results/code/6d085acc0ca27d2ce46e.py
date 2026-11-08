import numpy as np
from scipy.optimize import minimize, basinhopping
from scipy.special import roots_laguerre
import time
import math
import sys

def construct():
    start_time = time.time()
    # Deadline with safety margin
    deadline = start_time + 590

    m = 12
    n = 1

    # Precomputation: Transformation matrix from standard basis to Laguerre basis
    # P(x) = sum_{k=0}^{12} c_k L_k^{(1)}(x)
    # A[k,j] = coeff of x^j in L_k^{(1)}(x)
    A = np.zeros((13, 13))
    for k in range(13):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    weights = np.arange(13) + 1

    def compute_cost(log_roots):
        """
        Compute the cost function (upper bound on C4) given log roots.
        Cost = 1 / ||P||^2_Laguerre
        """
        # Time check
        if time.time() > deadline:
            return 1e20

        roots = np.exp(log_roots)
        
        # Validity check
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e20

        # Build polynomial P(x) = prod(x - r_i)
        # np.poly is efficient for small n
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]  # Ascending order
        
        # Compute Laguerre coefficients c
        c = A_T_inv @ p
        
        # Compute norm squared
        norm_sq = np.sum(c**2 * weights)
        
        if norm_sq <= 1e-10:
            return 1e20
        
        return 1.0 / norm_sq

    best_val = 1e20
    best_log_roots = None

    # Helper to update best
    def update_best(val, x):
        nonlocal best_val, best_log_roots
        if val < best_val:
            best_val = val
            best_log_roots = x.copy()

    # --- Phase 1: Smart Initial Guesses ---
    # The roots of Laguerre polynomials are often good candidates for this type of problem.
    # We generate a diverse set of starting points.
    
    initial_points = []
    
    # 1. Roots of L_12^(1)(x)
    try:
        l_roots, _ = roots_laguerre(12, 1)
        initial_points.append(np.log(l_roots))
        
        # Scaled versions
        for scale in [0.5, 2.0, 5.0]:
            initial_points.append(np.log(l_roots * scale))
    except Exception:
        pass

    # 2. Geometric Progressions
    # r_i = a * b^i
    for a in [1.0, 4.0, 10.0]:
        for b in [1.5, 2.0, 2.5]:
            r_geo = a * (b ** np.arange(12))
            initial_points.append(np.log(r_geo))
            
    # 3. Random Log-Uniform
    rng = np.random.RandomState(42)
    for _ in range(10):
        log_r = rng.uniform(np.log(0.1), np.log(1000.0), 12)
        initial_points.append(log_r)

    # --- Phase 2: Basin Hopping Search ---
    # Basin hopping uses a local optimizer to polish each jump.
    # This is structurally different from DE which uses population dynamics.
    
    # Local optimizer settings
    local_opts = {'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
    
    # Step function for basin hopping
    def step(x):
        # Random step in log-space
        # Use a moderate step size to escape local minima but not jump too far
        step_size = 0.5
        return x + rng.normal(0, step_size, size=x.shape)

    # We will run multiple independent basin hopping loops.
    # Each loop runs for a limited number of iterations to allow for many restarts.
    
    num_bh_runs = 50
    maxiter_bh = 30 # Iterations per basin hopping run
    
    for i in range(num_bh_runs):
        if time.time() > deadline - 10:
            break
            
        # Select an initial point
        # Cycle through the smart guesses, then use random
        if i < len(initial_points):
            x0 = initial_points[i]
        else:
            x0 = rng.uniform(np.log(0.1), np.log(1000.0), 12)
        
        try:
            res = basinhopping(
                compute_cost,
                x0,
                stepfunc=step,
                minimizer_kwargs={'method': 'L-BFGS-B', 'options': local_opts},
                niter=maxiter_bh,
                seed=i
            )
            
            # Update best
            if res.fun < best_val:
                update_best(res.fun, res.x)
                
            # Optional: If we found a very good point, we can do a final high-precision polish
            if best_val < 1e-10 and time.time() < deadline - 5:
                res_polish = minimize(
                    compute_cost, 
                    res.x, 
                    method='L-BFGS-B', 
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10}
                )
                if res_polish.fun < best_val:
                    update_best(res_polish.fun, res_polish.x)
                    
        except Exception as e:
            # If basin hopping fails, try a direct local minimize from x0
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

    # --- Phase 3: Final Polish ---
    if best_log_roots is not None:
        try:
            # One last high-precision polish
            res_final = minimize(
                compute_cost, 
                best_log_roots, 
                method='L-BFGS-B', 
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            if res_final.fun < best_val:
                best_val = res_final.fun
                best_log_roots = res_final.x
        except Exception:
            pass

    # Fallback
    if best_log_roots is None:
        best_log_roots = np.log(4.0 * (2.0 ** np.arange(12)))
        best_val = compute_cost(best_log_roots)

    # Convert back to roots
    best_roots = np.exp(best_log_roots)
    best_roots = np.sort(best_roots)
    best_roots = np.maximum(best_roots, 1e-10)
    
    return list(best_roots)

if __name__ == "__main__":
    roots = construct()
    print(roots)
