import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    n_vars = n_customers * n_facilities + n_facilities
    
    # Objective coefficients
    c = np.zeros(n_vars)
    c[:n_customers * n_facilities] = assign_cost.flatten()
    c[n_customers * n_facilities:] = open_cost
    
    # Integrality
    integrality = np.ones(n_vars)
    
    # Bounds
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    bounds = Bounds(lb, ub)
    
    constraints = []
    
    # 1. Each customer assigned to exactly one facility
    # sum_j x_ij = 1
    rows1 = np.repeat(np.arange(n_customers), n_facilities)
    cols1 = np.arange(n_customers * n_facilities)
    data1 = np.ones(n_customers * n_facilities)
    A1 = csr_matrix((data1, (rows1, cols1)), shape=(n_customers, n_vars))
    constraints.append(LinearConstraint(A1, 1, 1))
    
    # 2. Capacity constraints
    # sum_i demands_i * x_ij <= capacities_j
    rows2 = np.repeat(np.arange(n_facilities), n_customers)
    cols2 = np.arange(n_customers * n_facilities).reshape(n_facilities, n_customers).T.flatten() # This is wrong
    # Let's construct it properly
    # For each facility j, we have a row.
    # The non-zero elements are at columns i * n_facilities + j for i in 0..n_customers-1.
    rows2_list = []
    cols2_list = []
    data2_list = []
    for j in range(n_facilities):
        for i in range(n_customers):
            rows2_list.append(j)
            cols2_list.append(i * n_facilities + j)
            data2_list.append(demands[i])
    A2 = csr_matrix((data2_list, (rows2_list, cols2_list)), shape=(n_facilities, n_vars))
    constraints.append(LinearConstraint(A2, -np.inf, capacities))
    
    # 3. Linking constraints: x_ij <= y_j
    # x_ij - y_j <= 0
    rows3_list = []
    cols3_list = []
    data3_list = []
    for i in range(n_customers):
        for j in range(n_facilities):
            idx_x = i * n_facilities + j
            idx_y = n_customers * n_facilities + j
            row_idx = i * n_facilities + j
            rows3_list.append(row_idx)
            cols3_list.append(idx_x)
            data3_list.append(1)
            rows3_list.append(row_idx)
            cols3_list.append(idx_y)
            data3_list.append(-1)
    A3 = csr_matrix((data3_list, (rows3_list, cols3_list)), shape=(n_customers * n_facilities, n_vars))
    constraints.append(LinearConstraint(A3, -np.inf, 0))
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, options={"time_limit": 130})
        if res.x is not None:
            x_flat = res.x[:n_customers * n_facilities]
            x = x_flat.reshape((n_customers, n_facilities))
            x_rounded = np.round(x)
            if np.all(np.sum(x_rounded, axis=1) == 1):
                result = np.argmax(x_rounded, axis=1).astype(int)
                load = np.zeros(n_facilities)
                for i in range(n_customers):
                    load[result[i]] += demands[i]
                if np.all(load <= capacities + 1e-6):
                    return result
    except Exception:
        pass
    
    # Fallback: Greedy
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    order = np.argsort(-demands)
    for i in order:
        costs = assign_cost[i].copy()
        for f in range(n_facilities):
            if load[f] + demands[i] > capacities[f]:
                costs[f] = np.inf
        min_cost_idx = np.argmin(costs)
        if costs[min_cost_idx] == np.inf:
            min_cost_idx = 0
        result[i] = min_cost_idx
        load[min_cost_idx] += demands[i]
        
    return result
