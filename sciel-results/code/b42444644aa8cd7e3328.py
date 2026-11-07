# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def assign(demands, capacities, open_cost, assign_cost):
    """
    Solve the capacitated facility location problem using LP relaxation via linprog (HiGHS).
    The constraint matrix is totally unimodular, so the LP relaxation yields an integer solution.
    
    Parameters:
    - demands: array of shape (n_customers,) with integer demands
    - capacities: array of shape (n_facilities,) with integer capacities
    - open_cost: array of shape (n_facilities,) with float opening costs
    - assign_cost: array of shape (n_customers, n_facilities) with float assignment costs
    
    Returns:
    - array of shape (n_customers,) with integer facility indices
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin

    n_cust = len(demands)
    n_fac = len(capacities)

    # Decision variables:
    # x[i, f] = 1 if customer i is assigned to facility f (binary, but LP relaxation works due to TU)
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
    # 3. Linking constraint: x[i, f] <= y[f] for all i, f

    # Build constraint matrix
    # Number of constraints: n_cust (equality) + n_fac (capacity, inequality) + n_cust*n_fac (linking, inequality)
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

    # Variable bounds: all variables in [0, 1]
    bounds = [(0, 1)] * n_vars

    # Solve LP using HiGHS (default method for linprog)
    result = linprog(c, A_ub=A, b_ub=ub, A_eq=A[:n_cust], b_eq=lb[:n_cust], bounds=bounds, method='highs')

    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback to greedy if LP takes too long
        return _greedy_fallback(demands, capacities, open_cost, assign_cost)

    if not result.success:
        # Fallback to greedy if LP fails
        return _greedy_fallback(demands, capacities, open_cost, assign_cost)

    # Extract solution
    x = result.x[:n_cust * n_fac].reshape(n_cust, n_fac)
    assignment = np.zeros(n_cust, dtype=int)
    for i in range(n_cust):
        # Find the facility with x[i, f] closest to 1
        assignment[i] = np.argmax(x[i])

    # Verify validity: check capacity constraints
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[assignment[i]] += demands[i]
    
    if np.any(load > capacities):
        # If capacity is violated (shouldn't happen with TU, but just in case), use fallback
        return _greedy_fallback(demands, capacities, open_cost, assign_cost)

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
