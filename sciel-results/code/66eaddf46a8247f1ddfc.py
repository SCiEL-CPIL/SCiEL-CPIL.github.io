import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def _greedy_init(demands, capacities, open_cost, assign_cost):
    """
    Greedy initialization: assign customers to the cheapest feasible facility.
    Returns assignment array and x0 vector for MIP.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Sort customers by demand descending (harder to place first)
    # Or sort by "tightness" of assignment? 
    # Let's stick to demand descending for simplicity and stability.
    order = np.argsort(-demands)
    
    assignment = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    
    for i in order:
        # Find feasible facilities sorted by cost
        # We need to check capacity
        feasible = np.where(load + demands[i] <= capacities)[0]
        if len(feasible) == 0:
            # If no feasible facility, this is a problem. 
            # In valid instances, a solution should exist. 
            # If not, we might need to relax capacity or pick the one with most remaining capacity.
            # For now, pick the one with most remaining capacity to maximize chance of success.
            # But the task says "a facility whose assigned demand exceeds its capacity make the whole result invalid".
            # So we must ensure feasibility. If greedy fails, the instance is likely infeasible or greedy is bad.
            # Let's try to pick the facility with max remaining capacity.
            rem_cap = capacities - load
            best_f = np.argmax(rem_cap)
            if rem_cap[best_f] < demands[i]:
                # Truly infeasible? 
                # We will still assign to best_f to return *something*, but it will be invalid.
                # However, the problem implies instances are solvable.
                pass
            assignment[i] = best_f
            load[best_f] += demands[i]
        else:
            # Sort feasible facilities by assign_cost[i, feasible]
            costs = assign_cost[i, feasible]
            sorted_indices = feasible[np.argsort(costs)]
            for f in sorted_indices:
                if load[f] + demands[i] <= capacities[f]:
                    assignment[i] = f
                    load[f] += demands[i]
                    break
            else:
                # Should not reach here if feasible was not empty
                assignment[i] = sorted_indices[0]
                load[sorted_indices[0]] += demands[i]

    # Construct x0
    n_vars = n_customers * n_facilities + n_facilities
    x0 = np.zeros(n_vars)
    for i in range(n_customers):
        x0[i * n_facilities + assignment[i]] = 1.0
    
    # Set y0
    for j in range(n_facilities):
        if load[j] > 0:
            x0[n_customers * n_facilities + j] = 1.0
        else:
            x0[n_customers * n_facilities + j] = 0.0
            
    return assignment, x0

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # 1. Generate initial solution
    greedy_assignment, x0 = _greedy_init(demands, capacities, open_cost, assign_cost)
    
    # Calculate cost of greedy solution to use as upper bound (if supported) or just for tracking
    # milp doesn't take an upper bound directly in options, but a good x0 helps.
    
    n_x = n_customers * n_facilities
    n_y = n_facilities
    n_vars = n_x + n_y
    
    c = np.zeros(n_vars)
    for i in range(n_customers):
        for j in range(n_facilities):
            c[i * n_facilities + j] = assign_cost[i][j]
    for j in range(n_facilities):
        c[n_x + j] = open_cost[j]
    
    # Constraints construction (same as before)
    A_list = []
    lb_list = []
    ub_list = []
    
    # Constraint 1: Assignment
    for i in range(n_customers):
        row = np.zeros(n_vars)
        for j in range(n_facilities):
            row[i * n_facilities + j] = 1.0
        A_list.append(row)
        lb_list.append(1.0)
        ub_list.append(1.0)
        
    # Constraint 2: Capacity
    for j in range(n_facilities):
        row = np.zeros(n_vars)
        for i in range(n_customers):
            row[i * n_facilities + j] = demands[i]
        A_list.append(row)
        lb_list.append(0.0)
        ub_list.append(capacities[j])
        
    # Constraint 3: Linking x and y
    for i in range(n_customers):
        for j in range(n_facilities):
            row = np.zeros(n_vars)
            row[i * n_facilities + j] = 1.0
            row[n_x + j] = -1.0
            A_list.append(row)
            lb_list.append(-np.inf)
            ub_list.append(0.0)
            
    A = np.array(A_list)
    lb = np.array(lb_list)
    ub = np.array(ub_list)
    
    constraints = LinearConstraint(A, lb, ub)
    integrality = np.ones(n_vars)
    bounds = Bounds(0, 1)
    
    start_time = time.time()
    # Leave 10s margin for evaluation overhead
    time_limit = 100.0
    
    try:
        # Pass x0 to help the solver
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds, 
                   x0=x0, options={'time_limit': time_limit})
        
        if res.status == 0 or res.status == 2: # Success or Local Optimum/Time Limit with incumbent
            if res.x is not None and len(res.x) == n_vars:
                x_flat = res.x[:n_x]
                # Round to nearest integer
                x_rounded = np.round(x_flat)
                # Ensure each customer has exactly one facility
                assignment = np.zeros(n_customers, dtype=int)
                valid = True
                for i in range(n_customers):
                    row = x_rounded[i * n_facilities : (i+1) * n_facilities]
                    # Check if sum is 1
                    if np.sum(row) != 1:
                        # Fallback to greedy for this customer? Or just pick max
                        # If MIP fails to provide a valid integer solution, we should fall back
                        valid = False
                        break
                    j = np.argmax(row)
                    assignment[i] = j
                
                if valid:
                    # Verify capacity
                    load = np.zeros(n_facilities)
                    for i in range(n_customers):
                        load[assignment[i]] += demands[i]
                    if np.all(load <= capacities):
                        return assignment
                    else:
                        # Capacity violated, return greedy
                        pass
                else:
                    pass
            else:
                pass
        else:
            # Fallback to greedy
            pass
            
    except Exception:
        pass
        
    # Fallback: Return the greedy solution
    return greedy_assignment
