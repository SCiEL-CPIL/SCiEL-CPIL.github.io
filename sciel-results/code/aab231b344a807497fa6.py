"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, Bounds, LinearConstraint

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using MILP (scipy.optimize.milp with HiGHS).
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    n_customers = len(demands)
    n_facilities = len(capacities)

    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    # Variables:
    # x[i, f] for i in [0, n_customers), f in [0, n_facilities) - binary
    # y[f] for f in [0, n_facilities) - binary
    # Total variables: n_customers * n_facilities + n_facilities

    n_x = n_customers * n_facilities
    n_y = n_facilities
    n_vars = n_x + n_y

    # Objective: minimize sum(assign_cost[i,f] * x[i,f]) + sum(open_cost[f] * y[f])
    c = np.zeros(n_vars)
    for i in range(n_customers):
        for f in range(n_facilities):
            c[i * n_facilities + f] = assign_cost[i, f]
    for f in range(n_facilities):
        c[n_x + f] = open_cost[f]

    # Constraints:
    # 1. Each customer assigned to exactly one facility:
    #    sum_f x[i,f] = 1 for all i
    # 2. Capacity: sum_i demands[i] * x[i,f] <= capacities[f] * y[f] for all f
    #    => sum_i demands[i] * x[i,f] - capacities[f] * y[f] <= 0

    # Build A_ub and b_ub for inequality constraints (type <=)
    # And A_eq and b_eq for equality constraints (type ==)

    # Equality constraints: n_customers
    A_eq = np.zeros((n_customers, n_vars))
    b_eq = np.ones(n_customers)
    for i in range(n_customers):
        for f in range(n_facilities):
            A_eq[i, i * n_facilities + f] = 1.0

    # Inequality constraints: n_facilities (capacity)
    A_ub = np.zeros((n_facilities, n_vars))
    b_ub = np.zeros(n_facilities)
    for f in range(n_facilities):
        for i in range(n_customers):
            A_ub[f, i * n_facilities + f] = demands[i]
        A_ub[f, n_x + f] = -capacities[f]
        b_ub[f] = 0.0

    # Bounds: all variables in [0, 1]
    bounds = Bounds(lb=0.0, ub=1.0)

    # Integrality: all variables are binary (0 or 1)
    integrality = np.ones(n_vars)

    try:
        result = milp(
            c=c,
            constraints=[
                LinearConstraint(A_eq, lb=b_eq, ub=b_eq),
                LinearConstraint(A_ub, lb=-np.inf, ub=b_ub)
            ],
            integrality=integrality,
            bounds=bounds
        )
    except Exception:
        result = None

    if result is not None and result.success:
        x_flat = result.x
        # Extract x[i, f]
        assignment = np.zeros(n_customers, dtype=int)
        for i in range(n_customers):
            for f in range(n_facilities):
                if x_flat[i * n_facilities + f] > 0.5:
                    assignment[i] = f
                    break
            else:
                # Fallback: assign to cheapest feasible
                assignment[i] = int(np.argmin(assign_cost[i]))
        return assignment
    else:
        # Fallback: simple greedy
        assignment = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities, dtype=float)
        order = np.argsort(-demands)
        for i in order:
            best_f = -1
            best_c = float('inf')
            for f in range(n_facilities):
                if load[f] + demands[i] <= capacities[f]:
                    if assign_cost[i, f] < best_c:
                        best_c = assign_cost[i, f]
                        best_f = f
            if best_f == -1:
                best_f = int(np.argmin(assign_cost[i]))
            assignment[i] = best_f
            load[best_f] += demands[i]
        return assignment

# EVOLVE-BLOCK-END
