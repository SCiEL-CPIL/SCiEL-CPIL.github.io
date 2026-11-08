import numpy as np
from scipy.optimize import differential_evolution, minimize, dual_annealing, basinhopping
import time
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 590

    m = 12
    n = 1

    # Precomputation: Transformation matrix from standard basis to Laguerre basis
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k,j] = (-1)^j * C(k+1, j+1) / j!
    
    A = np.zeros((13, 13))
    for k in range(13):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    def compute_cost(log_roots):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return 1e20

        roots = np.exp(log_roots)
        
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e20

        # Compute coefficients of Q(x) = prod (1 - x/r_i)
        # Q(x) = sum_{k=0}^{12} q_k x^k
        # q_k = (-1)^k e_k(1/r_1, ..., 1/r_{12})
        
        inv_roots = 1.0 / roots
        # Use numpy.poly to get coefficients of prod (x - 1/r_i)?
        # No, we want prod (1 - x/r_i).
        # Let y = x. Terms are 1 and -x/r_i.
        # We can compute the elementary symmetric polynomials of inv_roots.
        # Or simply multiply the polynomials.
        
        # Start with 1
        q = np.array([1.0])
        for r in roots:
            # Multiply by (1 - x/r)
            # q_new[k] = q[k] - q[k-1]/r
            q_new = np.zeros(len(q) + 1)
            q_new[0] = q[0]
            for k in range(1, len(q)):
                q_new[k] = q[k] - q[k-1]/r
            q_new[len(q)] = -q[-1]/r
            q = q_new
        
        # q is now coefficients of Q(x) in ascending order: q[0] + q[1]x + ... + q[12]x^12
        # q[0] should be 1.
        
        # Compute Laguerre coefficients c
        c = A_T_inv @ q
        
        # Compute norm squared
        weights = np.arange(13) + 1
        norm_sq = np.sum(c**2 * weights)
        
        if norm_sq <= 1e-10:
            return 1e20
        
        return 1.0 / norm_sq

    best_val = 1e20
    best_log_roots = None

    # Bounds for log roots
    # Allow small roots (log < 0) and moderate large roots
    bounds_log = [(-20.0, 5.0)] * 12

    def update_best(val, x):
        nonlocal best_val, best_log_roots
        if val < best_val:
            best_val = val
            best_log_roots = x.copy()

    # Phase 1: Global Search
    # Since we expect small roots, initialize around log(0.01) = -4.6
    # DE with Sobol initialization
    
    de_seeds = [1, 2, 3, 4, 5, 6, 7, 8]
    
    for seed in de_seeds:
        if time.time() - start_time > deadline - 60:
            break
        try:
            res = differential_evolution(
                compute_cost, 
                bounds_log, 
                seed=seed, 
                maxiter=200, 
                popsize=20, 
                tol=1e-10, 
                polish=False,
                init='sobol'
            )
            if time.time() - start_time < deadline - 10:
                try:
                    res_polish = minimize(
                        compute_cost, 
                        res.x, 
                        method='L-BFGS-B', 
                        bounds=bounds_log, 
                        options={'maxiter': 500, 'ftol': 1e-15}
                    )
                    val = res_polish.fun
                    x = res_polish.x
                except:
                    val = res.fun
                    x = res.x
            else:
                val = res.fun
                x = res.x
            
            update_best(val, x)
        except Exception as e:
            pass

    # Phase 2: Local Search from specific points
    # Point 1: All roots small (e.g., 0.01)
    initial_points = [np.log(np.full(12, 0.01))]
    # Point 2: Geometric sequence of small roots
    r_small = 0.01 * (0.5 ** np.arange(12))
    initial_points.append(np.log(r_small))
    # Point 3: Random small roots
    rng = np.random.RandomState(123)
    for _ in range(10):
        log_r = rng.uniform(-20.0, 0.0, 12)
        initial_points.append(log_r)

    for i, init_x in enumerate(initial_points):
        if time.time() - start_time > deadline - 20:
            break
        try:
            res_lf = minimize(
                compute_cost, 
                init_x, 
                method='L-BFGS-B', 
                bounds=bounds_log, 
                options={'maxiter': 500, 'ftol': 1e-15}
            )
            update_best(res_lf.fun, res_lf.x)
        except:
            pass

    if best_log_roots is None:
        best_log_roots = np.log(np.full(12, 0.01))
        best_val = compute_cost(best_log_roots)

    best_roots = np.exp(best_log_roots)
    best_roots = np.sort(best_roots)
    best_roots = np.maximum(best_roots, 1e-10)
    
    return list(best_roots)

if __name__ == "__main__":
    roots = construct()
    print(roots)
