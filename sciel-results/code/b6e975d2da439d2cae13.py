import numpy as np
from scipy.optimize import minimize, basinhopping, differential_evolution
from scipy.special import roots_laguerre
import time
import math
import sys

def construct():
    start_time = time.time()
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

    def update_best(val, x):
        nonlocal best_val, best_log_roots
        if val < best_val:
            best_val = val
            best_log_roots = x.copy()

    # --- Phase 1: Smart Initial Guesses ---
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
    local_opts = {'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
    
    def step(x):
        step_size = 0.5
        return x + rng.normal(0, step_size, size=x.shape)

    num_bh_runs = 20
    maxiter_bh = 30

    for i in range(num_bh_runs):
        if time.time() > deadline - 10:
            break
            
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
            
            if res.fun < best_val:
                update_best(res.fun, res.x)
                
            if best_val < 1e-10 and time.time() < deadline - 5:
                res_polish = minimize(
                    compute_cost, 
                    res.x, 
                    method='L-BFGS-B', 
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10}
                )
                if res_polish.fun < best_val:
                    update_best(res_polish.fun, res_polish.x)
                    
        except Exception:
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

    # --- Phase 3: Differential Evolution (Global Search) ---
    # DE is good for escaping local minima that basin hopping might get stuck in.
    # We run it with a limited budget.
    
    bounds_log = [(0.0, 12.0)] * 12
    
    de_seeds = [1, 2, 3]
    for seed in de_seeds:
        if time.time() > deadline - 30:
            break
        try:
            res = differential_evolution(
                compute_cost, 
                bounds_log, 
                seed=seed, 
                maxiter=50, 
                popsize=10, 
                tol=1e-6, 
                polish=False
            )
            
            if time.time() < deadline - 10:
                res_polish = minimize(
                    compute_cost, 
                    res.x, 
                    method='L-BFGS-B', 
                    bounds=bounds_log, 
                    options={'maxiter': 500, 'ftol': 1e-15}
                )
                if res_polish.fun < best_val:
                    update_best(res_polish.fun, res_polish.x)
            else:
                if res.fun < best_val:
                    update_best(res.fun, res.x)
        except Exception:
            pass

    # --- Phase 4: Final Polish ---
    if best_log_roots is not None:
        try:
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

    best_roots = np.exp(best_log_roots)
    best_roots = np.sort(best_roots)
    best_roots = np.maximum(best_roots, 1e-10)
    
    return list(best_roots)

if __name__ == "__main__":
    roots = construct()
    print(roots)
