import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Check for scipy.optimize.milp
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        # Construct MILP
        
        # Variables: x[i, j] for i in range(n_customers), j in range(n_facilities)
        #             y[j] for j in range(n_facilities)
        # Total vars: n_customers * n_facilities + n_facilities
        
        n_vars = n_customers * n_facilities + n_facilities
        
        # Objective
        c = np.zeros(n_vars)
        # x costs
        c[:n_customers * n_facilities] = assign_cost.flatten()
        # y costs
        c[n_customers * n_facilities:] = open_cost
        
        # Equality constraints: sum_j x[i, j] = 1 for all i
        A_eq = np.zeros((n_customers, n_vars))
        b_eq = np.ones(n_customers)
        for i in range(n_customers):
            for j in range(n_facilities):
                A_eq[i, i * n_facilities + j] = 1.0
        
        # Inequality constraints: sum_i demand[i] * x[i, j] - capacity[j] * y[j] <= 0
        A_ub = np.zeros((n_facilities, n_vars))
        b_ub = np.zeros(n_facilities)
        for j in range(n_facilities):
            for i in range(n_customers):
                A_ub[j, i * n_facilities + j] = demands[i]
            A_ub[j, n_customers * n_facilities + j] = -capacities[j]
            
        # Bounds and integrality
        bounds = Bounds(0, 1)
        integrality = np.ones(n_vars)
        
        # Solve
        res = milp(
            c=c,
            constraints=[
                LinearConstraint(A_eq, lb=b_eq, ub=b_eq),
                LinearConstraint(A_ub, lb=-np.inf, ub=b_ub)
            ],
            integrality=integrality,
            bounds=bounds
        )
        
        if res.success:
            x_flat = res.x[:n_customers * n_facilities]
            x = x_flat.reshape(n_customers, n_facilities)
            # Find assigned facility for each customer
            result = np.argmax(x, axis=1).astype(int)
            # Verify validity
            load = np.zeros(n_facilities)
            for i in range(n_customers):
                f = result[i]
                load[f] += demands[i]
            if np.all(load <= capacities + 1e-6):
                return result
            else:
                # Fallback if somehow invalid (shouldn't happen with MILP)
                pass
                
    except Exception:
        pass
        
    # Fallback: Greedy + Local Search
    # This is a robust heuristic if MILP fails or is not available
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    
    # 1. Greedy Initialization
    order = np.argsort(-demands)
    for i in order:
        best_f = -1
        best_cost = np.inf
        for f in range(n_facilities):
            if load[f] + demands[i] <= capacities[f]:
                cost = assign_cost[i, f]
                if cost < best_cost:
                    best_cost = cost
                    best_f = f
        if best_f != -1:
            result[i] = best_f
            load[best_f] += demands[i]
        else:
            # If no facility has capacity, this instance might be infeasible?
            # The problem statement implies feasible instances.
            # If not, we must assign to the one with most remaining capacity or least penalty?
            # Assuming feasible, this shouldn't trigger.
            # Force assign to facility with max remaining capacity
            rem_cap = capacities - load
            best_f = np.argmax(rem_cap)
            result[i] = best_f
            load[best_f] += demands[i]
            
    # 2. Local Search (1-swap)
    # Vectorized 1-swap improvement
    improved = True
    time_limit = 10.0
    start_time = time.time()
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Calculate current cost
        current_cost = np.sum(assign_cost[np.arange(n_customers), result])
        # Calculate load
        load = np.zeros(n_facilities)
        for i in range(n_customers):
            load[result[i]] += demands[i]
            
        # Try moving each customer
        for i in range(n_customers):
            current_f = result[i]
            current_c = assign_cost[i, current_f]
            # Find best alternative
            # We need to ensure capacity allows the move
            # Temporarily remove i from current_f
            load[current_f] -= demands[i]
            
            # Find facility f' that minimizes assign_cost[i, f'] + (open_cost change if f' was empty)
            # Note: open_cost is only paid if facility is used.
            # Moving i to f' might open f' if it was empty, or close current_f if it becomes empty.
            
            # To keep it simple and fast, we can just check assignment cost first, then adjust for open costs if necessary.
            # However, open costs are significant.
            
            # Let's evaluate total cost change for each potential move.
            best_f = -1
            best_delta = 0 # We want to minimize cost, so delta < 0 is improvement
            
            # Current open costs
            used_facilities = set(result.tolist())
            
            for f in range(n_facilities):
                if f == current_f:
                    continue
                if load[f] + demands[i] <= capacities[f]:
                    # Calculate new cost
                    # Assignment cost change: assign_cost[i, f] - assign_cost[i, current_f]
                    delta_assign = assign_cost[i, f] - current_c
                    
                    # Open cost change
                    # Does current_f become empty?
                    new_load_current = load[current_f]
                    if new_load_current == 0 and current_f in used_facilities:
                        delta_open = -open_cost[current_f]
                    else:
                        delta_open = 0
                        
                    # Does f become non-empty?
                    if load[f] == 0 and f not in used_facilities:
                        delta_open += open_cost[f]
                        
                    total_delta = delta_assign + delta_open
                    if total_delta < best_delta:
                        best_delta = total_delta
                        best_f = f
            
            if best_f != -1:
                # Perform move
                result[i] = best_f
                load[best_f] += demands[i]
                improved = True
                
    return result
