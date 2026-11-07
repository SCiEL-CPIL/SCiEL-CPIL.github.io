"""Combined program: MIP (scipy.optimize.milp) + Local Search Refinement."""
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
    
    open_sum = 0.0
    used = np.zeros(n_facilities, dtype=bool)
    for i in range(n_customers):
        f = int(assignment[i])
        open_sum += 0.0 # placeholder, handled below
        used[f] = True
        # We sum assign cost separately for efficiency in loops, but here we do full sum
    # Actually, let's just sum it all properly
    assign_sum = np.sum(assign_cost[np.arange(n_customers), assignment])
    open_sum = np.sum(open_cost[used])
    return open_sum + assign_sum

def _local_search(demands, capacities, open_cost, assign_cost, assignment, time_limit=5.0):
    """
    Vectorized local search: try moving individual customers.
    Optimized for speed to run within a short time budget.
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
    max_no_improve = 8  # Allow more rounds before giving up
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try moving each customer in random order
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
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_result = result.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if not improved:
            if no_improve_count > max_no_improve:
                break
            # Random perturbation: swap two customers
            if np.random.random() < 0.5:  # Increase perturbation probability
                i1, i2 = np.random.choice(n_customers, 2, replace=False)
                f1, f2 = result[i1], result[i2]
                if f1 != f2:
                    # Check feasibility
                    if load[f1] - demands[i1] + demands[i2] <= capacities[f1] and \
                       load[f2] - demands[i2] + demands[i1] <= capacities[f2]:
                        load[f1] = load[f1] - demands[i1] + demands[i2]
                        load[f2] = load[f2] - demands[i2] + demands[i1]
                        result[i1], result[i2] = f2, f1
                        
    return best_result, best_cost

def _heuristic_fallback(demands, capacities, open_cost, assign_cost, deadline):
    """Fallback greedy + local search if MIP fails."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Simple greedy: sort by demand descending, assign to cheapest feasible
    order = np.argsort(-demands)
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    
    for i in order:
        d = demands[i]
        costs = assign_cost[i].copy()
        feasible = (load + d) <= capacities
        costs[~feasible] = np.inf
        
        if not np.any(np.isfinite(costs)):
            # Fallback: pick facility with most remaining capacity
            remaining = capacities - load
            f = np.argmin(remaining)
        else:
            f = np.argmin(costs)
        
        load[f] += d
        result[i] = f
    
    # Local search refinement
    ls_result, _ = _local_search(demands, capacities, open_cost, assign_cost, result, time_limit=10.0)
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
        # 1. Each customer assigned to exactly one facility:
        #    sum_f x[c,f] = 1 for each c
        # 2. Capacity: sum_c demands[c]*x[c,f] <= capacities[f]*y[f] for each f
        #    => sum_c demands[c]*x[c,f] - capacities[f]*y[f] <= 0
        # 3. y[f] >= x[c,f] for each c,f (if customer assigned to f, f must be open)
        #    => x[c,f] - y[f] <= 0

        # Build constraint matrix
        # Constraint 1: n_customers rows
        A1 = np.zeros((n_customers, n_vars))
        for c in range(n_customers):
            A1[c, c * n_facilities:(c + 1) * n_facilities] = 1.0

        # Constraint 2: n_facilities rows
        A2 = np.zeros((n_facilities, n_vars))
        for f in range(n_facilities):
            for c in range(n_customers):
                A2[f, c * n_facilities + f] = demands[c]
            A2[f, x_vars + f] = -capacities[f]

        # Constraint 3: n_customers * n_facilities rows
        A3 = np.zeros((n_customers * n_facilities, n_vars))
        for c in range(n_customers):
            for f in range(n_facilities):
                idx = c * n_facilities + f
                A3[idx, c * n_facilities + f] = 1.0
                A3[idx, x_vars + f] = -1.0

        # Combine constraints
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

        # Variable bounds: all in [0, 1]
        var_bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))

        # Integrality: all binary
        integrality = np.ones(n_vars)

        # Time limit for MIP: leave 20s for local search and safety
        mip_time_limit = max(10.0, deadline - time.time() - 20.0)
        
        result = milp(
            c=c_obj,
            constraints=constraints,
            integrality=integrality,
            bounds=var_bounds,
            options={'time_limit': mip_time_limit, 'mip_rel_gap': 0.05}
        )

        if result.status == 0 and result.x is not None:
            x_flat = result.x[:x_vars]
            # Reshape to (n_customers, n_facilities)
            x_matrix = x_flat.reshape((n_customers, n_facilities))
            # Assign to argmax (since we minimize cost, x should be 1 for assigned)
            mip_result = np.argmax(x_matrix, axis=1).astype(int)
            
            # Verify feasibility
            load_check = np.zeros(n_facilities)
            for i in range(n_customers):
                load_check[mip_result[i]] += demands[i]
            if np.any(load_check > capacities + 1e-6):
                mip_result = None # Infeasible, discard

    except Exception as e:
        # MIP failed, use fallback
        mip_result = None

    if mip_result is not None:
        # Refine with local search
        remaining_time = max(5.0, deadline - time.time() - 5.0)
        ls_result, _ = _local_search(demands, capacities, open_cost, assign_cost, mip_result, time_limit=remaining_time)
        return ls_result
    else:
        # Fallback to heuristic
        return _heuristic_fallback(demands, capacities, open_cost, assign_cost, deadline)
# EVOLVE-BLOCK-END
