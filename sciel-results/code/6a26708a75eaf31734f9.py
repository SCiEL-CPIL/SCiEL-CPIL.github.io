# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    # Set a deadline to ensure we return before the evaluator kills us
    # The evaluator allows 630s, we want to return well before.
    # Solving the LP for K=2000 should take < 10s.
    
    K = 2000
    
    # If K is 1, trivial case
    if K <= 1:
        return np.array([1], dtype=np.int64), np.array([0.0], dtype=np.float64)

    n_vars = K - 1
    ks = np.arange(2, K + 1)
    
    # Objective: Minimize sum( (log k / k) * f_k )
    # This maximizes -sum( f_k * log k / k )
    c = np.log(ks) / ks
    
    max_x = 10 * K
    
    # Construct constraint matrix A
    # A[i, j] corresponds to x = i+1, k = j+2
    # A[i, j] = floor(x/k) - 1/k
    
    # To avoid huge memory spikes, we can process in chunks or just allocate.
    # 20000 * 1999 * 8 bytes ~ 320 MB.
    # If memory is an issue, we could use a sparse solver, but linprog dense is usually fine for this size.
    
    try:
        A = np.zeros((max_x, n_vars), dtype=np.float64)
        for x in range(1, max_x + 1):
            # Vectorized computation for row x
            floors = np.floor_divide(x, ks)
            A[x-1, :] = floors.astype(np.float64) - (1.0 / ks)
            
        b = np.ones(max_x, dtype=np.float64)
        bounds = [(-10.0, 10.0)] * n_vars
        
        # Solve
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        
        if res.status == 0:
            f_vals = res.x
            domain = np.arange(1, K + 1, dtype=np.int64)
            values = np.zeros(K, dtype=np.float64)
            values[1:] = f_vals
            return domain, values
        else:
            # Fallback
            pass
    except Exception:
        pass

    # Fallback to a smaller K or heuristic if LP fails
    # Try K=100
    K_fb = 100
    n_vars_fb = K_fb - 1
    ks_fb = np.arange(2, K_fb + 1)
    c_fb = np.log(ks_fb) / ks_fb
    max_x_fb = 10 * K_fb
    
    try:
        A_fb = np.zeros((max_x_fb, n_vars_fb), dtype=np.float64)
        for x in range(1, max_x_fb + 1):
            floors = np.floor_divide(x, ks_fb)
            A_fb[x-1, :] = floors.astype(np.float64) - (1.0 / ks_fb)
        b_fb = np.ones(max_x_fb, dtype=np.float64)
        bounds_fb = [(-10.0, 10.0)] * n_vars_fb
        
        res_fb = linprog(c_fb, A_ub=A_fb, b_ub=b_fb, bounds=bounds_fb, method='highs')
        if res_fb.status == 0:
            f_vals_fb = res_fb.x
            domain_fb = np.arange(1, K_fb + 1, dtype=np.int64)
            values_fb = np.zeros(K_fb, dtype=np.float64)
            values_fb[1:] = f_vals_fb
            return domain_fb, values_fb
    except Exception:
        pass

    # Final fallback: Heuristic
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values
# EVOLVE-BLOCK-END
