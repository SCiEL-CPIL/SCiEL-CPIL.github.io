import numpy as np
import time
import random
from scipy.optimize import milp, LinearConstraint, Bounds

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    demands = np.array(demands, dtype=float)
    capacities = np.array(capacities, dtype=float)
    open_cost = np.array(open_cost, dtype=float)
    assign_cost = np.array(assign_cost, dtype=float)
    
    def compute_cost(result):
        load = np.zeros(n_fac)
        fac_count = np.zeros(n_fac, dtype=int)
        for i, f in enumerate(result):
            load[f] += demands[i]
            fac_count[f] += 1
        if np.any(load > capacities + 1e-9):
            return float('inf')
        return np.sum(open_cost[fac_count > 0]) + np.sum(assign_cost[np.arange(n_cust), result])

    def greedy_init(order_func, randomize=False):
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        order = np.argsort(order_func(demands))
        
        for i in order:
            fac_order = np.argsort(assign_cost[i])
            if randomize:
                perturbed = assign_cost[i] + np.random.uniform(0, 1) * 10
                fac_order = np.argsort(perturbed)
            
            assigned = False
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break
            
            if not assigned:
                remaining = capacities - load
                f = np.argmax(remaining)
                if remaining[f] >= demands[i]:
                    load[f] += demands[i]
                    result[i] = f
                else:
                    return None
            
        return result

    # Generate a quick heuristic solution first as a safety net
    candidates = []
    init_g1 = greedy_init(lambda d: -d)
    if init_g1 is not None:
        candidates.append(init_g1)
        
    init_g2 = greedy_init(lambda d: d)
    if init_g2 is not None:
        candidates.append(init_g2)
        
    best_fallback = None
    if candidates:
        best_fallback = min(candidates, key=compute_cost)
    else:
        # Last resort: just fill sequentially
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        for i in range(n_cust):
            for f in range(n_fac):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break
        best_fallback = result

    # --- Enhanced Local Search ---
    def local_search(result, max_iters=10000):
        """
        Perform local search to improve the solution.
        Moves:
        1. Single customer reassignment (if capacity allows and cost decreases)
        2. Swap two customers between different facilities (if capacity allows and cost decreases)
        """
        result = result.copy()
        load = np.zeros(n_fac)
        for i, f in enumerate(result):
            load[f] += demands[i]
            
        # Precompute assignment costs for each customer to all facilities
        # assign_cost is already (n_cust, n_fac)
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 10:
            improved = False
            iters += 1
            
            # Shuffle order to avoid bias
            order = np.random.permutation(n_cust)
            
            for i in order:
                current_f = result[i]
                current_assign_cost = assign_cost[i, current_f]
                
                # Try moving customer i to another facility
                # Sort other facilities by assign_cost
                other_facs = np.argsort(assign_cost[i])
                
                for new_f in other_facs:
                    if new_f == current_f:
                        continue
                    
                    # Check capacity
                    if load[new_f] + demands[i] <= capacities[new_f] + 1e-9:
                        # Calculate delta
                        delta_assign = assign_cost[i, new_f] - current_assign_cost
                        delta_open = 0.0
                        
                        # If current facility becomes empty, save open cost
                        if load[current_f] == demands[i]:
                            delta_open -= open_cost[current_f]
                        # If new facility was empty, pay open cost
                        if load[new_f] == 0:
                            delta_open += open_cost[new_f]
                        
                        if delta_assign + delta_open < -1e-9:
                            # Apply move
                            load[current_f] -= demands[i]
                            load[new_f] += demands[i]
                            result[i] = new_f
                            improved = True
                            break # Restart inner loop for this customer? Or continue?
                            # Usually breaking and restarting the outer loop is safer for convergence
                            # But for speed, we can just continue to next customer
                            
            # Try swaps
            if time.time() > deadline - 15:
                break
                
            for i in range(n_cust):
                if time.time() > deadline - 15:
                    break
                f1 = result[i]
                # Find a potential swap partner
                # To be efficient, we might just try a random subset or specific heuristics
                # Here we try a limited number of swaps per iteration to keep it fast
                for _ in range(5):
                    j = np.random.randint(n_cust)
                    if i == j:
                        continue
                    f2 = result[j]
                    if f1 == f2:
                        continue
                    
                    # Check if swap is feasible
                    # load[f1] - d[i] + d[j] <= cap[f1]
                    # load[f2] - d[j] + d[i] <= cap[f2]
                    if (load[f1] - demands[i] + demands[j] <= capacities[f1] + 1e-9 and
                        load[f2] - demands[j] + demands[i] <= capacities[f2] + 1e-9):
                        
                        # Calculate delta
                        delta_assign = (assign_cost[i, f2] + assign_cost[j, f1]) - (assign_cost[i, f1] + assign_cost[j, f2])
                        delta_open = 0.0
                        
                        # Open cost changes are tricky with swaps
                        # If f1 becomes empty after removing i, and j moves there? No, j moves to f1.
                        # So f1 is not empty.
                        # If f2 becomes empty after removing j, and i moves there? No, i moves to f2.
                        # So f2 is not empty.
                        # However, if f1 had only i, and we move i to f2, f1 becomes empty.
                        # But we are swapping i and j. So i leaves f1, j enters f1.
                        # So f1 is not empty.
                        # Similarly f2 is not empty.
                        # So delta_open is 0 for swaps between non-empty facilities.
                        # Wait, what if f1 had only i? Then load[f1] == demands[i].
                        # After swap: load[f1] = demands[j]. Still not empty.
                        # So open cost doesn't change for swaps.
                        
                        if delta_assign < -1e-9:
                            # Apply swap
                            load[f1] -= demands[i]
                            load[f1] += demands[j]
                            load[f2] -= demands[j]
                            load[f2] += demands[i]
                            result[i] = f2
                            result[j] = f1
                            improved = True
                            break # Restart swap search for this i? Or just break inner loop?
                            
        return result

    # Run local search on the best fallback
    if time.time() < deadline - 20:
        best_fallback = local_search(best_fallback, max_iters=5000)

    # --- 2. Exact MILP Formulation ---
    # We will try MILP if we have enough time left.
    # The current program uses a large constraint matrix. We will keep that but ensure it's efficient.
    
    n_vars = n_cust * n_fac + n_fac
    
    # Objective: Minimize sum(assign_cost[i, j] * x[i, j]) + sum(open_cost[j] * y[j])
    c_obj = np.zeros(n_vars)
    for i in range(n_cust):
        for j in range(n_fac):
            c_obj[i * n_fac + j] = assign_cost[i, j]
    for j in range(n_fac):
        c_obj[n_cust * n_fac + j] = open_cost[j]
        
    # Integrality: All variables are binary (0 or 1)
    integrality = np.ones(n_vars)
    
    # Bounds: All variables are in [0, 1]
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    
    # Constraints
    n_rows = n_cust + n_fac + n_cust * n_fac
    A = np.zeros((n_rows, n_vars))
    lb_cons = np.zeros(n_rows)
    ub_cons = np.zeros(n_rows)
    
    # 1. Assignment Equality: sum_j x[i, j] = 1
    row_idx = 0
    for i in range(n_cust):
        for j in range(n_fac):
            A[row_idx, i * n_fac + j] = 1.0
        lb_cons[row_idx] = 1.0
        ub_cons[row_idx] = 1.0
        row_idx += 1
        
    # 2. Capacity: sum_i demands[i] x[i, j] <= capacities[j]
    for j in range(n_fac):
        for i in range(n_cust):
            A[row_idx, i * n_fac + j] = demands[i]
        lb_cons[row_idx] = -np.inf
        ub_cons[row_idx] = capacities[j]
        row_idx += 1
        
    # 3. Linking: x[i, j] <= y[j]  =>  x[i, j] - y[j] <= 0
    for i in range(n_cust):
        for j in range(n_fac):
            A[row_idx, i * n_fac + j] = 1.0
            A[row_idx, n_cust * n_fac + j] = -1.0
            lb_cons[row_idx] = -np.inf
            ub_cons[row_idx] = 0.0
            row_idx += 1
            
    constraints = LinearConstraint(A, lb_cons, ub_cons)
    bounds = Bounds(lb, ub)
    
    milp_solution = None
    try:
        if time.time() < deadline - 5:
            # Use HiGHS solver
            # We can try to use the heuristic solution as a warm start?
            # scipy.optimize.milp doesn't directly support warm start in the simple interface.
            # But we can rely on presolve and the solver's heuristics.
            res = milp(c=c_obj, constraints=constraints, integrality=integrality, bounds=bounds, 
                       options={'presolve': True, 'time_limit': max(0, deadline - time.time() - 2)})
            
            if res.status == 0 or res.status == 1: # Optimal or Feasible
                if hasattr(res, 'x') and res.x is not None:
                    x_sol = res.x
                    x_assign = x_sol[:n_cust * n_fac].reshape(n_cust, n_fac)
                    # Round to nearest integer
                    x_assign_rounded = np.round(x_assign).astype(int)
                    
                    # Verify feasibility
                    valid = True
                    load = np.zeros(n_fac)
                    for i in range(n_cust):
                        assigned_facs = np.where(x_assign_rounded[i] == 1)[0]
                        if len(assigned_facs) != 1:
                            valid = False
                            break
                        f = assigned_facs[0]
                        load[f] += demands[i]
                    
                    if np.any(load > capacities + 1e-9):
                        valid = False
                        
                    if valid:
                        result = np.zeros(n_cust, dtype=int)
                        for i in range(n_cust):
                            result[i] = np.where(x_assign_rounded[i] == 1)[0][0]
                        milp_solution = result
    except Exception:
        milp_solution = None

    # Choose the best solution
    if milp_solution is not None:
        cost_milp = compute_cost(milp_solution)
        cost_fallback = compute_cost(best_fallback)
        
        if cost_milp <= cost_fallback:
            return milp_solution
        else:
            return best_fallback
    else:
        return best_fallback
