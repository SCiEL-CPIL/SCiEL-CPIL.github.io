# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds


def assign(demands, capacities, open_cost, assign_cost):
    """
    Solve the capacitated facility location problem using Mixed Integer Programming.
    
    Parameters:
    - demands: array of shape (n_customers,) with integer demands
    - capacities: array of shape (n_facilities,) with integer capacities
    - open_cost: array of shape (n_facilities,) with float opening costs
    - assign_cost: array of shape (n_customers, n_facilities) with float assignment costs
    
    Returns:
    - array of shape (n_customers,) with integer facility indices
    """
    start_time = time.time()
    deadline = start_time + 130.0  # 130s safety margin out of 150s
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Decision variables:
    # x[i, f] = 1 if customer i is assigned to facility f (binary)
    # y[f] = 1 if facility f is open (binary)
    #
    # Total variables: n_cust * n_fac + n_fac
    n_vars = n_cust * n_fac + n_fac
    
    # Objective: minimize sum of open costs + assignment costs
    c = np.zeros(n_vars)
    for i in range(n_cust):
        for f in range(n_fac):
            c[i * n_fac + f] = assign_cost[i, f]
    for f in range(n_fac):
        c[n_cust * n_fac + f] = open_cost[f]
    
    # Constraints:
    # 1. Each customer assigned to exactly one facility:
    #    sum_f x[i, f] = 1 for each i
    # 2. Capacity constraint: sum_i demands[i] * x[i, f] <= capacities[f] * y[f] for each f
    # 3. Linking constraint: x[i, f] <= y[f] for all i, f (if customer assigned to f, f must be open)
    
    # Build constraint matrix
    # Number of constraints: n_cust + n_fac + n_cust * n_fac
    n_constraints = n_cust + n_fac + n_cust * n_fac
    
    A = np.zeros((n_constraints, n_vars))
    lb = np.zeros(n_constraints)
    ub = np.zeros(n_constraints)
    
    row = 0
    # Constraint 1: each customer assigned to exactly one facility
    for i in range(n_cust):
        for f in range(n_fac):
            A[row, i * n_fac + f] = 1.0
        lb[row] = 1.0
        ub[row] = 1.0
        row += 1
    
    # Constraint 2: capacity constraint
    # sum_i demands[i] * x[i, f] - capacities[f] * y[f] <= 0
    for f in range(n_fac):
        for i in range(n_cust):
            A[row, i * n_fac + f] = demands[i]
        A[row, n_cust * n_fac + f] = -capacities[f]
        lb[row] = -np.inf
        ub[row] = 0.0
        row += 1
    
    # Constraint 3: linking constraint x[i, f] <= y[f]
    # x[i, f] - y[f] <= 0
    for i in range(n_cust):
        for f in range(n_fac):
            A[row, i * n_fac + f] = 1.0
            A[row, n_cust * n_fac + f] = -1.0
            lb[row] = -np.inf
            ub[row] = 0.0
            row += 1
    
    # Variable bounds and integrality
    integrality = np.zeros(n_vars)
    for i in range(n_cust):
        for f in range(n_fac):
            integrality[i * n_fac + f] = 1.0  # binary
    for f in range(n_fac):
        integrality[n_cust * n_fac + f] = 1.0  # binary
    
    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    
    constraint = LinearConstraint(A, lb, ub)
    
    # Solve MIP
    result = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback to greedy if MIP takes too long
        return _greedy_fallback(demands, capacities, open_cost, assign_cost)
    
    if not result.success:
        # Fallback to greedy if MIP fails
        return _greedy_fallback(demands, capacities, open_cost, assign_cost)
    
    # Extract solution
    x = result.x[:n_cust * n_fac].reshape(n_cust, n_fac)
    assignment = np.zeros(n_cust, dtype=int)
    for i in range(n_cust):
        # Find the facility with x[i, f] closest to 1
        assignment[i] = np.argmax(x[i])
    
    return assignment


def _greedy_fallback(demands, capacities, open_cost, assign_cost):
    """Greedy fallback: largest demand first, cheapest feasible facility."""
    result = np.zeros(len(demands), dtype=int)
    load = np.zeros(len(capacities))
    for i in np.argsort(-demands):
        for f in np.argsort(assign_cost[i]):
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
    return result
# EVOLVE-BLOCK-END
