# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 30.0
    effective_time_limit = time_limit - safety_margin

    # The problem is linear in f(k) once the normalization is accounted for.
    # We formulate an LP over a small dense domain [1, K_max] to find the optimal
    # feasible solution that maximizes the objective a = -sum_k f(k) log(k) / k.
    
    K_max = 200  # Dense domain size. Small enough to solve quickly with scipy, large enough to capture structure.
    
    # Prepare data for LP
    # Variables: f_1, ..., f_K
    # Normalization: S = sum_{k=1}^K f_k / k. New f_1' = f_1 - S.
    # Objective: a = -sum_{k=1}^K f_k' log(k) / k
    #            = -sum_{k=2}^K f_k log(k)/k - f_1' log(1)/1
    #            = -sum_{k=2}^K f_k log(k)/k - (f_1 - S) * 0
    #            = -sum_{k=2}^K f_k log(k)/k
    # Wait, log(1) = 0, so f_1' term vanishes in objective.
    # So we maximize a = -sum_{k=2}^K f_k log(k)/k.
    # Subject to:
    #   -10 <= f_k <= 10 for all k (clipping is part of eval, but here we optimize pre-clipped values? 
    #   No, "values are clipped to [-10, 10]. Then f(1) is replaced..."
    #   So the inputs f_k are clipped. We can assume f_k in [-10, 10] effectively.
    #   If we output values outside, they get clipped. Since objective is linear, optimum is on boundary.
    #   So we can restrict f_k in [-10, 10].
    #
    # Constraint: M(x) <= 1.0001 for all x in [1, 10*K_max].
    # M(x) = sum_{k=1}^K f_k' floor(x/k)
    #      = sum_{k=1}^K (f_k - delta_k S) floor(x/k)  where delta_k = 1 if k==1 else 0?
    #      Actually f_1' = f_1 - S. f_k' = f_k for k>1.
    #      M(x) = (f_1 - S) * floor(x/1) + sum_{k=2}^K f_k floor(x/k)
    #           = f_1 * floor(x) - S * floor(x) + sum_{k=2}^K f_k floor(x/k)
    #           = sum_{k=1}^K f_k floor(x/k) - S * floor(x)
    #
    # S = sum_{k=1}^K f_k / k.
    # So M(x) = sum_{k=1}^K f_k floor(x/k) - floor(x) * sum_{k=1}^K f_k / k
    #         = sum_{k=1}^K f_k ( floor(x/k) - floor(x)/k )
    #
    # Let c_k(x) = floor(x/k) - floor(x)/k.
    # Constraint: sum_{k=1}^K c_k(x) f_k <= 1.0001 for all x in [1, 10*K_max].
    #
    # We want to maximize a = -sum_{k=2}^K f_k log(k)/k.
    # Let c_k = -log(k)/k for k>1, c_1 = 0.
    # Maximize c^T f
    # Subject to:
    #   A f <= b  where A_{x,k} = c_k(x), b_x = 1.0001
    #   -10 <= f_k <= 10
    
    from scipy.optimize import linprog
    
    K = K_max
    n_vars = K
    
    # Objective coefficients
    c_obj = np.zeros(K)
    for k in range(1, K+1):
        if k > 1:
            c_obj[k-1] = -np.log(k) / k
        else:
            c_obj[k-1] = 0.0
            
    # Maximize c_obj^T f  => Minimize -c_obj^T f
    c_min = -c_obj
    
    # Constraints A_ub f <= b_ub
    # x ranges from 1 to 10*K
    x_max = 10 * K
    num_constraints = x_max
    A_ub = np.zeros((num_constraints, n_vars))
    b_ub = np.full(num_constraints, 1.0001)
    
    # Precompute log_k_div_k for speed?
    # Loop over x and k
    # A_ub[x-1, k-1] = floor(x/k) - floor(x)/k
    
    # This loop might be slow if K=200, x_max=2000. 200*2000 = 400k ops. Fine.
    # But constructing the matrix in Python loop is slow.
    # Let's vectorize or use efficient loops.
    
    # Actually, c_k(x) = floor(x/k) - (x // 1) / k. Wait floor(x) is x for integer x.
    # c_k(x) = x//k - x/k.
    
    # Let's build A_ub efficiently.
    # For each k, c_k(x) is a step function.
    # We can fill columns.
    for k in range(1, K+1):
        # x from 1 to x_max
        # c = x//k - x/k
        # We can compute this for all x at once
        xs = np.arange(1, x_max+1)
        c_vals = (xs // k) - (xs / k)
        A_ub[:, k-1] = c_vals
        
    # Bounds
    bounds = [(-10.0, 10.0)] * n_vars
    
    # Solve
    try:
        res = linprog(c_min, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status != 0:
            # Fallback to seed
            domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
            values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
            return domain, values
            
        f_opt = res.x
        # Clip just in case
        f_opt = np.clip(f_opt, -10, 10)
        
        # Verify feasibility quickly
        # Check M(x) for all x
        # M(x) = sum f_k (x//k - x/k)
        # This is the same as A_ub @ f_opt <= 1.0001
        # But A_ub was built with f_k.
        # Let's check max M(x).
        M_vals = A_ub @ f_opt
        if np.any(M_vals > 1.0001 + 1e-6):
             # Infeasible? Return seed
             domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
             values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
             return domain, values
             
        # Construct domain and values
        # Include all k where f_opt is not close to 0?
        # Or just return all K?
        # Domain must be distinct positive integers.
        domain = np.arange(1, K+1, dtype=np.int64)
        values = f_opt.astype(np.float64)
        
        return domain, values
    except Exception as e:
        # Fallback
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

# EVOLVE-BLOCK-END
