import math
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# Precompute 12!
T12 = math.factorial(12)

# Prime factorization of 12!: 2^10 * 3^5 * 5^2 * 7^1 * 11^1
primes = [2, 3, 5, 7, 11]
target_exponents = [10, 5, 2, 1, 1]

def generate_divisors_with_exponents():
    """
    Generate all divisors of 12! along with their prime exponents.
    Returns a list of tuples: (divisor_value, exponent_vector)
    """
    divisors = [(1, [0, 0, 0, 0, 0])]
    for i, (p, e_target) in enumerate(zip(primes, target_exponents)):
        new_divisors = []
        p_power = 1
        for _ in range(e_target + 1):
            for d, exps in divisors:
                new_exp = exps.copy()
                new_exp[i] += _ # This logic is slightly off, need to be careful with iteration
                # Actually, standard generation is better:
            # Let's redo this properly to avoid confusion
            pass
        
    # Correct generation
    divisors = [1]
    exp_vectors = [[0, 0, 0, 0, 0]]
    
    for i, (p, e_target) in enumerate(zip(primes, target_exponents)):
        new_divisors = []
        new_exp_vectors = []
        p_power = 1
        for _ in range(e_target + 1):
            for d, ev in zip(divisors, exp_vectors):
                new_exp = ev.copy()
                new_exp[i] = _
                new_divisors.append(d * p_power)
                new_exp_vectors.append(new_exp)
            p_power *= p
        divisors = new_divisors
        exp_vectors = new_exp_vectors
        
    result = []
    for d, ev in zip(divisors, exp_vectors):
        if d > 12:
            result.append((d, ev))
            
    return result

# Generate candidate divisors
# We filter out 1 and anything <= 12
raw_divisors = generate_divisors_with_exponents()
# Sort by value for consistent indexing
raw_divisors.sort(key=lambda x: x[0])
divisor_values = [d for d, _ in raw_divisors]
divisor_exponents = np.array([ev for _, ev in raw_divisors])

n_divisors = len(divisor_values)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin before 90s kill
    
    # If no divisors (shouldn't happen), return trivial
    if n_divisors == 0:
        return [T12]
    
    # ILP Formulation:
    # Variables:
    # x_0 ... x_{n-1} : binary, selection of divisor
    # x_n : continuous, T (max factor)
    # Total variables: n + 1
    
    # Objective: Minimize T
    # c = [0, 0, ..., 0, 1]
    
    # Constraints:
    # 1. Sum of exponents matches target for each prime
    #    For each prime i: sum_j (x_j * exp_j_i) = target_exp_i
    # 2. T >= d_j * x_j for all j
    #    T - sum_j (d_j * x_j) ... wait, T >= d_j * x_j is linear: T - d_j*x_j >= 0
    # 3. Sum x_j <= 12
    # 4. Sum x_j >= 1
    
    # Let's build the constraint matrix A_ub and b_ub for A_ub @ x <= b_ub
    # Or use LinearConstraint with lb and ub
    
    # Variables: [x_0, ..., x_{n-1}, T]
    
    # Constraint 1: Prime exponents
    # A_exp is 5 x (n+1)
    # Row i: divisor_exponents[:, i] for x part, 0 for T
    A_exp = np.zeros((5, n_divisors + 1))
    for i in range(5):
        A_exp[i, :n_divisors] = divisor_exponents[:, i]
    b_exp = target_exponents
    
    # Constraint 2: T >= d_j * x_j  =>  T - d_j * x_j >= 0  =>  -T + d_j * x_j <= 0
    # This is n constraints.
    A_T = np.zeros((n_divisors, n_divisors + 1))
    for j in range(n_divisors):
        A_T[j, j] = divisor_values[j]  # Coefficient for x_j
        A_T[j, n_divisors] = -1        # Coefficient for T
    b_T = np.zeros(n_divisors)
    
    # Constraint 3: Sum x_j <= 12
    A_sum = np.zeros((1, n_divisors + 1))
    A_sum[0, :n_divisors] = 1
    b_sum = np.array([12])
    
    # Constraint 4: Sum x_j >= 1  =>  -Sum x_j <= -1
    A_sum_min = np.zeros((1, n_divisors + 1))
    A_sum_min[0, :n_divisors] = -1
    b_sum_min = np.array([-1])
    
    # Combine constraints
    # We can use scipy.optimize.milp with LinearConstraint objects or just A_ub, b_ub
    # Let's use A_ub, b_ub format for simplicity if bounds are handled separately
    
    A_ub = np.vstack([A_T, A_sum, A_sum_min])
    b_ub = np.concatenate([b_T, b_sum, b_sum_min])
    
    # Equality constraints for exponents
    A_eq = A_exp
    b_eq = b_exp
    
    # Bounds
    # x_j in [0, 1] for j < n
    # T in [13, T12] (T must be at least 13, at most 12!)
    lb = np.zeros(n_divisors + 1)
    ub = np.ones(n_divisors + 1)
    ub[n_divisors] = T12  # T can be up to 12!
    lb[n_divisors] = 13   # T must be > 12
    
    # Integrality
    integrality = np.zeros(n_divisors + 1)
    integrality[:n_divisors] = 1  # x_j are binary
    # T is continuous
    
    # Objective
    c = np.zeros(n_divisors + 1)
    c[n_divisors] = 1  # Minimize T
    
    try:
        res = milp(
            c=c,
            constraints=(
                LinearConstraint(A_ub, -np.inf, b_ub),
                LinearConstraint(A_eq, b_eq, b_eq)
            ),
            integrality=integrality,
            bounds=Bounds(lb, ub)
        )
        
        if res.status == 0:  # Optimal
            # Extract solution
            x = res.x
            selected_divisors = []
            for j in range(n_divisors):
                if x[j] > 0.5:
                    selected_divisors.append(divisor_values[j])
            
            # Verify solution
            if selected_divisors:
                prod = 1
                for d in selected_divisors:
                    prod *= d
                if prod == T12 and len(selected_divisors) <= 12 and all(d > 12 for d in selected_divisors):
                    # Distinctness is guaranteed by binary variables on unique divisors
                    # Sort for consistency (not required but nice)
                    return sorted(selected_divisors)
                else:
                    # Fallback if ILP returned weird result
                    return [T12]
            else:
                return [T12]
        else:
            # Infeasible or error, fallback
            return [T12]
            
    except Exception as e:
        # If ILP fails, fallback to a quick heuristic or trivial
        # Given the structure, [T12] is always a valid solution (1 factor, >12, product 12!)
        # But we want to minimize max. 
        # A simple heuristic: Try to find a 2-factor decomposition close to sqrt(T12)
        # But [T12] is safe.
        return [T12]

# Quick test to ensure it runs
if __name__ == "__main__":
    res = construct()
    print(res)
