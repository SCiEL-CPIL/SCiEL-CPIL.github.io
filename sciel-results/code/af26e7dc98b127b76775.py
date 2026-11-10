# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f
    """
    t_start = time.time()
    time_budget = 600  # seconds
    safety_margin = 30  # seconds to spare
    
    # We will try to solve the LP for small K
    # K=10 is small enough for LP
    K = 10
    domain = np.arange(1, K + 1, dtype=np.int64)
    
    # Variables: f_2, ..., f_K
    # f_1 is implicit: f_1 = - sum_{k=2}^K f(k)/k
    # Objective: maximize - sum_{k=1}^K f(k) log(k)/k
    # = - [ f_1 * 0 + sum_{k=2}^K f(k) log(k)/k ]
    # = - sum_{k=2}^K f(k) log(k)/k
    # c = - [ log(2)/2, log(3)/3, ..., log(K)/K ]
    # We want to minimize c^T x
    
    c_obj = -np.array([np.log(k)/k for k in range(2, K+1)])
    
    # Constraints: M(x) <= 1.0001 for x in 1..10K
    # M(x) = f_1 x + sum_{k=2}^K f(k) floor(x/k)
    # f_1 = - sum_{k=2}^K f(k)/k
    # M(x) = -x sum_{k=2}^K f(k)/k + sum_{k=2}^K f(k) floor(x/k)
    # M(x) = sum_{k=2}^K f(k) (floor(x/k) - x/k)
    # M(x) = - sum_{k=2}^K f(k) {x/k}
    
    # A_ub * x <= b_ub
    # For each x, - sum_{k=2}^K {x/k} f(k) <= 1.0001
    # sum_{k=2}^K {x/k} f(k) >= -1.0001
    # - sum_{k=2}^K {x/k} f(k) <= 1.0001
    
    N_x = 10 * K
    A_ub = np.zeros((N_x, K-1))
    b_ub = np.full(N_x, 1.0001)
    
    for i, x in enumerate(range(1, N_x + 1)):
        for j, k in enumerate(range(2, K+1)):
            A_ub[i, j] = - (x/k - np.floor(x/k)) # - {x/k}
            
    # Bounds: -10 <= f(k) <= 10
    bounds = [(-10, 10)] * (K-1)
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            f_2_to_K = res.x
            f_1 = -np.sum(f_2_to_K / np.arange(2, K+1))
            values = np.concatenate(([f_1], f_2_to_K))
            # Clip to [-10, 10] just in case
            values = np.clip(values, -10, 10)
            return domain, values
        else:
            # Fallback
            pass
    except Exception:
        pass
    
    # Fallback to simple heuristic
    domain = np.array([1, 2], dtype=np.int64)
    values = np.array([1.0001, -2.0002], dtype=np.float64)
    return domain, values

# EVOLVE-BLOCK-END
