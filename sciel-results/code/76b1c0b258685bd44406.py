import numpy as np
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    K = 50
    max_x = 10 * K
    
    # Objective: minimize sum(f_k * ln(k) / k)
    # We want to maximize -sum(f_k * ln(k) / k), so minimize sum(f_k * ln(k) / k)
    c = np.array([np.log(k) / k for k in range(1, K + 1)])
    
    # Constraints: M(x) <= 1 for x in 1..max_x
    # A_ub[i, j] = floor((i+1) / (j+1))
    A_ub = np.zeros((max_x, K))
    for i in range(max_x):
        x = i + 1
        for j in range(K):
            k = j + 1
            A_ub[i, j] = x // k
    b_ub = np.ones(max_x)
    
    # Equality constraint: sum(f_k / k) = 0
    A_eq = np.array([1.0 / k for k in range(1, K + 1)])
    b_eq = np.array([0.0])
    
    # Bounds
    bounds = [(-10.0, 10.0) for _ in range(K)]
    
    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if res.status == 0:
        values = res.x
        # Clip to [-10, 10] just in case of numerical issues, though bounds handle it
        values = np.clip(values, -10, 10)
        
        # Verify feasibility strictly
        domain = np.arange(1, K + 1, dtype=np.int64)
        # Recalculate M(x) to ensure no floating point errors violated constraint
        M = np.zeros(max_x + 1)
        for i, k in enumerate(domain):
            f_k = values[i]
            if f_k != 0:
                for x in range(1, max_x + 1):
                    M[x] += f_k * (x // k)
        
        if np.max(M[1:]) <= 1.0001:
             return domain, values
             
    # Fallback to simple 2-term solution if LP fails or is invalid
    domain = np.array([1, 2], dtype=np.int64)
    values = np.array([1.0, -2.0], dtype=np.float64)
    return domain, values
