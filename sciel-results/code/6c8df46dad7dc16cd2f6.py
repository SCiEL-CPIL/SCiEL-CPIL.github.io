"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from math import factorial

def construct():
    N = 12
    target = factorial(N)
    
    # Get divisors > N
    divisors = []
    i = 1
    while i * i <= target:
        if target % i == 0:
            if i > N:
                divisors.append(i)
            if i != target // i and target // i > N:
                divisors.append(target // i)
        i += 1
    
    divisors = sorted(list(set(divisors)))
    n_divs = len(divisors)
    
    if n_divs == 0:
        return [target]
        
    # Prime factorization of 12!
    primes = [2, 3, 5, 7, 11]
    target_exponents = [10, 5, 2, 1, 1]
    
    # Calculate exponents for each divisor
    exp_matrix = np.zeros((n_divs, len(primes)), dtype=int)
    for idx, d in enumerate(divisors):
        temp_d = d
        for p_idx, p in enumerate(primes):
            while temp_d % p == 0:
                exp_matrix[idx, p_idx] += 1
                temp_d //= p
                
    # Variables: x_0 ... x_{n_divs-1} (binary), M (continuous)
    n_vars = n_divs + 1
    
    # Objective: minimize M
    c = np.zeros(n_vars)
    c[-1] = 1.0
    
    # Constraints
    # 1. Exponent equalities
    A_exp = np.zeros((len(primes), n_vars))
    for j in range(len(primes)):
        A_exp[j, :n_divs] = exp_matrix[:, j]
        
    lb_exp = target_exponents
    ub_exp = target_exponents
    
    # 2. Upper bound constraints: d_i * x_i - M <= 0
    A_ub = np.zeros((n_divs, n_vars))
    for i in range(n_divs):
        A_ub[i, i] = divisors[i]
        A_ub[i, -1] = -1.0
        
    lb_ub = np.zeros(n_divs)
    ub_ub = np.zeros(n_divs)
    
    # Combine
    A = np.vstack([A_exp, A_ub])
    lb = np.concatenate([lb_exp, lb_ub])
    ub = np.concatenate([ub_exp, ub_ub])
    
    cons = LinearConstraint(A, lb, ub)
    
    # Bounds
    # x_i in [0, 1], M in [13, target]
    bounds = Bounds(lb=np.zeros(n_vars), ub=np.concatenate([np.ones(n_divs), [target]]))
    bounds.lb[-1] = 13
    
    # Integrality
    integrality = np.zeros(n_vars)
    integrality[:n_divs] = 1
    
    res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds)
    
    if res.status == 0:
        x = res.x
        selected_divisors = []
        for i in range(n_divs):
            if x[i] > 0.5:
                selected_divisors.append(divisors[i])
                
        if selected_divisors:
            prod = 1
            for d in selected_divisors:
                prod *= d
            if prod == target and all(d > N for d in selected_divisors) and len(set(selected_divisors)) == len(selected_divisors):
                return selected_divisors
    
    # Fallback
    return [target]

# EVOLVE-BLOCK-END
