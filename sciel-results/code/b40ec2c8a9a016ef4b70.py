"""
Facility location solver using MIP (scipy.optimize.milp) with enhanced local search.
Strategy:
1. Solve MIP for a strong initial solution
2. Apply multi-pass local search with 1-swap and 2-swap moves
3. Use time budget carefully to stay within limits
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_cost(assignment, demands, capacities, open_cost, assign_cost):
    """Compute total cost: assignment + opening costs."""
    n_customers = len(assignment)
    n_facilities = len(capacities)
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        f = int(assignment[i])
        if f < 0 or f >= n_facilities:
            return np.inf
        load[f] += demands[i]
    
    # Check capacity
    if np.any(load > capacities + 1e-9):
        return np.inf
    
    assign_sum = np.sum(assign_cost[np.arange(n_customers), assignment])
    used = np.zeros(n_facilities, dtype=bool)
    for i in range(n_customers):
        used[int(assignment[i])] = True
    open_sum = np.sum(open_cost[used])
    return open_sum + assign_sum

def _local_search_enhanced(demands, capacities, open_cost, assign_cost, assignment, time_limit=10.0):
    """
    Enhanced local search with 1-swap and 2-swap moves.
    Uses vectorized operations for speed.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    result = assignment.copy().astype(int)
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        load[result[i]] += demands[i]
    
    current_cost = _compute_cost(result, demands, capacities, open_cost, assign_cost)
    if np.isinf(current_cost):
        return result, current_cost
        
    best_cost = current_cost
    best_result = result.copy()
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 5
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Pass 1: 1-swap moves (move individual customers)
        for i in np.random.permutation(n_customers):
            if time.time() - start_time > time_limit:
                break
            
            d = demands[i]
            current_f = result[i]
            
            # Vectorized cost computation for all facilities
            delta_assign = assign_cost[i] - assign_cost[i, current_f]
            
            # Delta open cost logic
            load_after_source = load[current_f] - d
            closes_source = (load_after_source == 0)
            
            # For each facility f, check if it would be opened
            opens_dest = (load == 0)
            
            # delta_open for each facility
            delta_open = np.zeros(n_facilities)
            if closes_source:
                delta_open[:] -= open_cost[current_f]
            # Add open cost for facilities that are currently empty
            delta_open[opens_dest] += open_cost[opens_dest]
            
            # Total delta
            total_delta = delta_assign + delta_open
            
            # Mask infeasible: capacity violation or same facility
            feasible = (load + d) <= capacities
            feasible[current_f] = False  # Don't move to same facility
            
            if not np.any(feasible):
                continue
            
            # Find best improvement
            total_delta_masked = np.where(feasible, total_delta, np.inf)
            best_f = np.argmin(total_delta_masked)
            best_delta = total_delta_masked[best_f]
            
            if best_delta < -1e-9:
                # Perform move
                load[current_f] -= d
                load[best_f] += d
                result[i] = best_f
                current_cost += best_delta
                improved = True
        
        # Pass 2: 2-swap moves (swap two customers between facilities)
        if time.time() - start_time < time_limit * 0.7:
            # Try random pairs of customers
            n_pairs = min(200, n_customers * (n_customers - 1) // 2)
            for _ in range(n_pairs):
                if time.time() - start_time > time_limit:
                    break
                
                i1, i2 = np.random.choice(n_customers, 2, replace=False)
                f1, f2 = result[i1], result[i2]
                
                if f1 == f2:
                    continue
                
                d1, d2 = demands[i1], demands[i2]
                
                # Check feasibility after swap
                new_load_f1 = load[f1] - d1 + d2
                new_load_f2 = load[f2] - d2 + d1
                
                if new_load_f1 > capacities[f1] or new_load_f2 > capacities[f2]:
                    continue
                
                # Compute delta cost
                delta_assign = (assign_cost[i1, f2] + assign_cost[i2, f1]) - \
                               (assign_cost[i1, f1] + assign_cost[i2, f2])
                
                # Open cost changes
                delta_open = 0.0
                if new_load_f1 == 0 and load[f1] > 0:
                    # f1 would close
                    delta_open -= open_cost[f1]
                if new_load_f2 == 0 and load[f2] > 0:
                    # f2 would close
                    delta_open -= open_cost[f2]
                if new_load_f1 > 0 and load[f1] == 0:
                    # f1 would open
                    delta_open += open_cost[f1]
                if new_load_f2 > 0 and load[f2] == 0:
                    # f2 would open
                    delta_open += open_cost[f2]
                
                total_delta = delta_assign + delta_open
                
                if total_delta < -1e-9:
                    # Perform swap
                    load[f1] = new_load_f1
                    load[f2] = new_load_f2
                    result[i1], result[i2] = f2, f1
                    current_cost += total_delta
                    improved = True
        
        if current_cost < best_cost - 1e-9:
            best_cost = current_cost
            best_result = result.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if not improved:
            if no_improve_count > max_no_improve:
                break
            # Random perturbation to escape local optima
            if np.random.random() < 0.3:
                # Swap two random customers
                i1, i2 = np.random.choice(n_customers, 2, replace=False)
                f1, f2 = result[i1], result[i2]
                if f1 != f2:
                    d1, d2 = demands[i1], demands[i2]
                    if load[f1] - d1 + d2 <= capacities[f1] and \
                       load[f2] - d2 + d1 <= capacities[f2]:
                        load[f1] = load[f1] - d1 + d2
                        load[f2] = load[f2] - d2 + d1
                        result[i1], result[i2] = f2, f1
                        
    return best_result, best_cost

def _heuristic_fallback(demands, capacities, open_cost, assign_cost, deadline):
    """Fallback greedy + local search if MIP fails."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Improved greedy: sort by demand descending, assign to cheapest feasible
    # with consideration for open costs
    order = np.argsort(-demands)
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    open_status = np.zeros(n_facilities, dtype=bool)
    
    for i in order:
        d = demands[i]
        costs = assign_cost[i].copy()
        feasible = (load + d) <= capacities
        
        # Add open cost penalty for facilities not yet open
        for f in range(n_facilities):
            if not open_status[f]:
                costs[f] += open_cost[f]
        
        costs[~feasible] = np.inf
        
        if not np.any(np.isfinite(costs)):
            # Fallback: pick facility with most remaining capacity
            remaining = capacities - load
            f = np.argmin(remaining)
        else:
            f = np.argmin(costs)
        
        load[f] += d
        open_status[f] = True
        result[i] = f
    
    # Local search refinement with more time
    remaining = max(5.0, deadline - time.time() - 2.0)
    ls_result, _ = _local_search_enhanced(demands, capacities, open_cost, assign_cost, result, time_limit=remaining)
    return ls_result

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Try MIP first
    mip_result = None
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import csr_matrix

        # Variables:
        # x[c, f] = 1 if customer c assigned to facility f (n_customers * n_facilities)
        # y[f] = 1 if facility f is open (n_facilities)
        # Total vars: n_customers * n_facilities + n_facilities

        n_vars = n_customers * n_facilities + n_facilities
        x_vars = n_customers * n_facilities
        y_vars = n_facilities

        # Objective: minimize sum_c,f assign_cost[c,f]*x[c,f] + sum_f open_cost[f]*y[f]
        c_obj = np.zeros(n_vars)
        for c in range(n_customers):
            for f in range(n_facilities):
                c_obj[c * n_facilities + f] = assign_cost[c, f]
        for f in range(n_facilities):
            c_obj[x_vars + f] = open_cost[f]

        # Constraints:
        # 1. Each customer assigned to exactly one facility
        # 2. Capacity constraint
        # 3. Openness constraint

        n_cons = n_customers + n_facilities + n_customers * n_facilities
        A = np.zeros((n_cons, n_vars))
        lb = np.zeros(n_cons)
        ub = np.zeros(n_cons)

        # Type 1: equality (assignment)
        for c in range(n_customers):
            A[c, c * n_facilities:(c + 1) * n_facilities] = 1.0
            lb[c] = 1.0
            ub[c] = 1.0

        # Type 2: capacity (<= 0)
        for f in range(n_facilities):
            row = n_customers + f
            for c in range(n_customers):
                A[row, c * n_facilities + f] = demands[c]
            A[row, x_vars + f] = -capacities[f]
            lb[row] = -np.inf
            ub[row] = 0.0

        # Type 3: openness (x[c,f] - y[f] <= 0)
        for c in range(n_customers):
            for f in range(n_facilities):
                row = n_customers + n_facilities + c * n_facilities + f
                A[row, c * n_facilities + f] = 1.0
                A[row, x_vars + f] = -1.0
                lb[row] = -np.inf
                ub[row] = 0.0

        constraints = LinearConstraint(A, lb, ub)
        var_bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
        integrality = np.ones(n_vars)

        # Time limit for MIP: leave 20s for local search and safety
        mip_time_limit = max(15.0, deadline - time.time() - 20.0)
        
        result = milp(
            c=c_obj,
            constraints=constraints,
            integrality=integrality,
            bounds=var_bounds,
            options={'time_limit': mip_time_limit, 'mip_rel_gap': 0.01}
        )

        if result.status == 0 and result.x is not None:
            x_flat = result.x[:x_vars]
            x_matrix = x_flat.reshape((n_customers, n_facilities))
            mip_result = np.argmax(x_matrix, axis=1).astype(int)
            
            # Verify feasibility
            load_check = np.zeros(n_facilities)
            for i in range(n_customers):
                load_check[mip_result[i]] += demands[i]
            if np.any(load_check > capacities + 1e-6):
                mip_result = None

    except Exception:
        mip_result = None

    if mip_result is not None:
        # Refine with enhanced local search
        remaining_time = max(10.0, deadline - time.time() - 5.0)
        ls_result, _ = _local_search_enhanced(demands, capacities, open_cost, assign_cost, mip_result, time_limit=remaining_time)
        return ls_result
    else:
        # Fallback to heuristic
        return _heuristic_fallback(demands, capacities, open_cost, assign_cost, deadline)
# EVOLVE-BLOCK-END
