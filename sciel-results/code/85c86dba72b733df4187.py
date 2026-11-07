import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy import sparse

def assign(demands, capacities, open_cost, assign_cost):
    n_cust = len(demands)
    n_fac = len(capacities)
    n_x = n_cust * n_fac
    n_y = n_fac
    n_vars = n_x + n_y
    
    # Objective
    c = np.zeros(n_vars)
    c[:n_x] = assign_cost.flatten()
    c[n_x:] = open_cost
    
    # Bounds
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    
    # Integrality
    integrality = np.ones(n_vars)
    
    # Constraints
    # We will build sparse rows
    
    # 1. Assignment: sum_j x_ij = 1
    # Indices for x variables
    # For each i, we have n_fac ones at positions i*n_fac to (i+1)*n_fac - 1
    data_assign = []
    row_assign = []
    col_assign = []
    for i in range(n_cust):
        for j in range(n_fac):
            data_assign.append(1.0)
            row_assign.append(i)
            col_assign.append(i * n_fac + j)
            
    # 2. Capacity: sum_i d_i x_ij <= cap_j
    data_cap = []
    row_cap = []
    col_cap = []
    for j in range(n_fac):
        for i in range(n_cust):
            data_cap.append(demands[i])
            row_cap.append(n_cust + j) # Offset rows
            col_cap.append(i * n_fac + j)
            
    # 3. Linking: x_ij - y_j <= 0
    data_link = []
    row_link = []
    col_link = []
    for i in range(n_cust):
        for j in range(n_fac):
            data_link.append(1.0)
            row_link.append(n_cust + n_fac + i * n_fac + j)
            col_link.append(i * n_fac + j)
            
            data_link.append(-1.0)
            row_link.append(n_cust + n_fac + i * n_fac + j)
            col_link.append(n_x + j)
            
    # Combine into single sparse matrix
    all_data = data_assign + data_cap + data_link
    all_row = row_assign + row_cap + row_link
    all_col = col_assign + col_cap + col_link
    
    n_constraints = n_cust + n_fac + n_cust * n_fac
    A = sparse.coo_matrix((all_data, (all_row, all_col)), shape=(n_constraints, n_vars)).tocsr()
    
    lb_con = np.zeros(n_constraints)
    ub_con = np.zeros(n_constraints)
    
    # Assignment: = 1
    lb_con[:n_cust] = 1.0
    ub_con[:n_cust] = 1.0
    
    # Capacity: <= cap
    lb_con[n_cust:n_cust+n_fac] = -np.inf
    ub_con[n_cust:n_cust+n_fac] = capacities
    
    # Linking: <= 0
    lb_con[n_cust+n_fac:] = -np.inf
    ub_con[n_cust+n_fac:] = 0.0
    
    constraint = LinearConstraint(A, lb_con, ub_con)
    bounds = Bounds(lb, ub)
    
    res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
    
    if not res.success:
        # Fallback
        return _greedy(demands, capacities, assign_cost)
        
    x = res.x
    result = np.zeros(n_cust, dtype=int)
    # Extract x_ij
    x_mat = x[:n_x].reshape(n_cust, n_fac)
    # argmax gives the assigned facility (since sum is 1 and binary)
    result = np.argmax(x_mat, axis=1)
    
    return result

def _greedy(demands, capacities, assign_cost):
    result = np.zeros(len(demands), dtype=int)
    load = np.zeros(len(capacities))
    for i in np.argsort(-demands):
        for f in np.argsort(assign_cost[i]):
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
    return result
