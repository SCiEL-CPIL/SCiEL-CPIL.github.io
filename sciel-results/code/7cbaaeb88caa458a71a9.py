"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csc_matrix


def assign(demands, capacities, open_cost, assign_cost):
    """
    Solve the capacitated facility location problem using MILP (HiGHS via scipy).
    
    Variables:
      x[i, f] = 1 if customer i is assigned to facility f (binary)
      y[f]    = 1 if facility f is open (binary)
    
    Objective: minimize sum_f open_cost[f]*y[f] + sum_{i,f} assign_cost[i,f]*x[i,f]
    
    Constraints:
      1. Each customer assigned to exactly one facility: sum_f x[i,f] = 1 for all i
      2. Capacity: sum_i demands[i]*x[i,f] <= capacities[f]*y[f] for all f
      3. Linking: x[i,f] <= y[f] for all i,f (implied by capacity if demands>0, but explicit is safer)
         Actually, if a customer is assigned to f, f must be open. Since demands[i] >= 5 > 0,
         capacity constraint implies y[f] must be 1 if any x[i,f]=1. But to be safe and standard,
         we can rely on capacity constraints alone since demands are positive.
    
    We use a time budget of 100 seconds to leave margin before the 120s/150s limit.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Deadline guard
    start_time = time.time()
    time_limit = 100  # seconds
    
    # Flatten assignment variables: x[i*n_facilities + f]
    # Open variables: y[f] at offset n_customers * n_facilities
    n_x = n_customers * n_facilities
    n_y = n_facilities
    n_vars = n_x + n_y
    
    # Objective coefficients
    c = np.zeros(n_vars)
    # x[i, f] cost
    for i in range(n_customers):
        for f in range(n_facilities):
            c[i * n_facilities + f] = assign_cost[i, f]
    # y[f] cost
    for f in range(n_facilities):
        c[n_x + f] = open_cost[f]
    
    # Bounds
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    bounds = Bounds(lb, ub)
    
    # Integrality: all binary
    integrality = np.ones(n_vars)
    
    # Constraints:
    # 1. Assignment constraints: for each customer i, sum_f x[i,f] = 1
    #    A1 has shape (n_customers, n_vars)
    # 2. Capacity constraints: for each facility f, sum_i demands[i]*x[i,f] - capacities[f]*y[f] <= 0
    #    A2 has shape (n_facilities, n_vars)
    
    A_rows = []
    A_cols = []
    A_data = []
    
    # Assignment constraints (equality)
    for i in range(n_customers):
        for f in range(n_facilities):
            A_rows.append(i)
            A_cols.append(i * n_facilities + f)
            A_data.append(1.0)
    
    # Capacity constraints (inequality: <= 0)
    # sum_i demands[i] * x[i,f] - capacities[f] * y[f] <= 0
    for f in range(n_facilities):
        for i in range(n_customers):
            A_rows.append(n_customers + f)
            A_cols.append(i * n_facilities + f)
            A_data.append(float(demands[i]))
        # y[f] term
        A_rows.append(n_customers + f)
        A_cols.append(n_x + f)
        A_data.append(-float(capacities[f]))
    
    A = csc_matrix((A_data, (A_rows, A_cols)), shape=(n_customers + n_facilities, n_vars))
    
    # Lower and upper bounds for constraints
    lb_con = np.zeros(n_customers + n_facilities)
    ub_con = np.zeros(n_customers + n_facilities)
    
    # Assignment constraints: equality to 1
    lb_con[:n_customers] = 1.0
    ub_con[:n_customers] = 1.0
    
    # Capacity constraints: <= 0
    lb_con[n_customers:] = -np.inf
    ub_con[n_customers:] = 0.0
    
    constraint = LinearConstraint(A, lb_con, ub_con)
    
    # Solve
    try:
        result = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
        
        if result.status == 0 and result.x is not None:
            x_flat = result.x[:n_x]
            # Threshold at 0.5 to get binary assignments
            x_matrix = x_flat.reshape(n_customers, n_facilities)
            assignments = np.argmax(x_matrix, axis=1)
            
            # Verify validity
            load = np.zeros(n_facilities)
            for i in range(n_customers):
                load[assignments[i]] += demands[i]
            
            for f in range(n_facilities):
                if load[f] > capacities[f] + 1e-6:
                    # Fallback to greedy
                    return _greedy_fallback(demands, capacities, assign_cost)
            
            return assignments.astype(int)
        else:
            # Fallback to greedy
            return _greedy_fallback(demands, capacities, assign_cost)
            
    except Exception:
        # Fallback to greedy
        return _greedy_fallback(demands, capacities, assign_cost)


def _greedy_fallback(demands, capacities, assign_cost):
    """Greedy fallback: largest demand first, cheapest feasible facility."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    
    for i in np.argsort(-demands):
        sorted_f = np.argsort(assign_cost[i])
        for f in sorted_f:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
        else:
            # Should not happen if capacities are sufficient, but assign to first available
            for f in range(n_facilities):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break
    
    return result
# EVOLVE-BLOCK-END
