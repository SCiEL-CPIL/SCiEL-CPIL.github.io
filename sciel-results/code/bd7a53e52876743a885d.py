"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import factorial
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds

    N = 12
    target = factorial(N)
    
    # Prime factors of 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    target_exponents = [10, 5, 2, 1, 1]
    
    # Generate all divisors of target that are > N
    divisors = []
    exponents_list = []
    
    for e1 in range(11):
        for e2 in range(6):
            for e3 in range(3):
                for e4 in range(2):
                    for e5 in range(2):
                        d = (2 ** e1) * (3 ** e2) * (5 ** e3) * (7 ** e4) * (11 ** e5)
                        if d > N:
                            divisors.append(d)
                            exponents_list.append([e1, e2, e3, e4, e5])
                            
    n_vars = len(divisors)
    
    # MILP Setup
    # Variables: x_0 ... x_{n_vars-1} (binary), M (continuous)
    # Total variables: n_vars + 1
    
    # Objective: Minimize M
    c = np.zeros(n_vars + 1)
    c[-1] = 1.0
    
    # Constraints:
    # 1. Equality constraints for prime exponents
    A_eq = np.zeros((len(primes), n_vars + 1))
    for i in range(n_vars):
        for j in range(len(primes)):
            A_eq[j, i] = exponents_list[i][j]
            
    b_eq = np.array(target_exponents)
    cons_eq = LinearConstraint(A_eq, b_eq, b_eq)
    
    # 2. Inequality constraints: M >= x_i * d_i  =>  d_i * x_i - M <= 0
    A_ub = np.zeros((n_vars, n_vars + 1))
    for i in range(n_vars):
        A_ub[i, i] = divisors[i]
        A_ub[i, -1] = -1.0
        
    ub = np.zeros(n_vars)
    lb = np.full(n_vars, -np.inf)
    cons_ub = LinearConstraint(A_ub, lb, ub)
    
    constraints = [cons_eq, cons_ub]
    
    # Variable bounds
    bounds = Bounds(
        lb=np.array([0.0] * n_vars + [13.0]),
        ub=np.array([1.0] * n_vars + [np.inf])
    )
    
    # Integrality
    integrality = np.array([1] * n_vars + [0])
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0 and res.success:
            x = res.x[:-1]
            
            selected = []
            for i in range(n_vars):
                if x[i] > 0.5:
                    selected.append(divisors[i])
            
            if selected:
                # Verify product
                prod = 1
                for s in selected:
                    prod *= s
                    
                if prod == target and all(s > N for s in selected):
                    return selected
                    
    except Exception:
        pass
    
    # Fallback
    return [target]

# EVOLVE-BLOCK-END
