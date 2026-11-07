import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    best_result = None
    best_score = float('inf')
    
    def evaluate(result):
        load = np.zeros(n_facilities)
        for i, f in enumerate(result):
            load[f] += demands[i]
        if np.any(load > capacities + 1e-9):
            return float('inf')
        
        open_mask = np.zeros(n_facilities, dtype=bool)
        for f in result:
            open_mask[f] = True
        
        open_sum = np.sum(open_cost[open_mask])
        assign_sum = np.sum(assign_cost[np.arange(n_customers), result])
        return open_sum + assign_sum
    
    # Greedy Initialization with optional randomization
    def greedy_init(randomize=False, seed=None):
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
        
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        is_open = np.zeros(n_facilities, dtype=bool)
        
        # Sort customers by demand descending
        order = np.argsort(-demands)
        
        for i in order:
            d = demands[i]
            # Feasible facilities
            feasible = load + d <= capacities
            
            if not np.any(feasible):
                f = np.argmin(assign_cost[i])
                result[i] = f
                load[f] += d
                is_open[f] = True
                continue
            
            # Calculate marginal costs
            marginal_costs = assign_cost[i].copy()
            closed_mask = ~is_open
            marginal_costs[closed_mask] += open_cost[closed_mask]
            
            # Mask infeasible
            marginal_costs[~feasible] = np.inf
            
            if randomize:
                # Pick from top 5 with some randomness
                sorted_indices = np.argsort(marginal_costs)
                top_k = sorted_indices[:min(5, n_facilities)]
                # Weighted random selection favoring lower cost
                weights = 1.0 / (marginal_costs[top_k] + 1e-9)
                weights /= weights.sum()
                f = top_k[rng.choice(len(top_k), p=weights)]
            else:
                f = np.argmin(marginal_costs)
            
            result[i] = f
            load[f] += d
            is_open[f] = True
            
        return result, load, is_open
    
    # Lagrangian Relaxation approach
    def lagrangian_relaxation(max_iter=200, step_size=None):
        """
        Uses Lagrangian relaxation for capacity constraints.
        For each customer i, the relaxed cost for facility j is:
        assign_cost[i,j] + lambda_j * demands[i] + (open_cost[j] if facility j is "opened")
        
        We use a subgradient method to update lambda (dual variables).
        The subgradient for lambda_j is (load_j - capacity_j) where load_j is the
        total demand assigned to facility j in the relaxed solution.
        """
        if step_size is None:
            step_size = 5.0
        
        # Initialize dual variables to 0
        lambdas = np.zeros(n_facilities)
        
        best_relaxed_result = None
        best_relaxed_score = float('inf')
        
        # Track best feasible solution found during the process
        best_feasible_result = None
        best_feasible_score = float('inf')
        
        for it in range(max_iter):
            if time.time() > deadline - 2:
                break
            
            # Solve the relaxed problem: for each customer, pick facility minimizing
            # assign_cost[i,j] + lambda_j * demands[i]
            # Note: We ignore open_cost in the relaxed problem for simplicity,
            # or we can include a fixed charge. Let's include it via a heuristic:
            # We'll compute the relaxed assignment, then determine which facilities are used.
            
            # Compute relaxed costs: shape (n_customers, n_facilities)
            # relaxed_cost[i, j] = assign_cost[i, j] + lambdas[j] * demands[i]
            relaxed_costs = assign_cost + np.outer(demands, lambdas)
            
            # For each customer, pick the best facility
            relaxed_result = np.argmin(relaxed_costs, axis=1)
            
            # Compute load
            load = np.zeros(n_facilities)
            for i, f in enumerate(relaxed_result):
                load[f] += demands[i]
            
            # Determine which facilities are "open" (have at least one customer)
            is_open = load > 0
            
            # Compute the actual objective for this relaxed solution
            # (this is an upper bound on the dual, but we use it for tracking)
            # Actually, the dual objective is: sum_i min_j (assign_cost[i,j] + lambda_j * demands[i])
            # But we need a feasible solution to evaluate. Let's try to repair.
            
            # Repair heuristic: 
            # 1. Start with relaxed_result
            # 2. If any facility exceeds capacity, move customers out to least loaded facilities
            # 3. Add open costs for facilities that become open
            
            repaired_result = relaxed_result.copy()
            load_rep = np.zeros(n_facilities)
            for i, f in enumerate(repaired_result):
                load_rep[f] += demands[i]
            
            # Fix capacity violations
            # Find facilities that exceed capacity
            violated = load_rep > capacities + 1e-9
            
            # Sort customers by demand descending for better repair
            order = np.argsort(-demands)
            
            for i in order:
                f_old = repaired_result[i]
                if load_rep[f_old] <= capacities[f_old] + 1e-9:
                    continue
                
                d = demands[i]
                # Find a facility with enough capacity
                # Prefer facilities with lowest marginal cost
                feasible = load_rep + d <= capacities
                
                if np.any(feasible):
                    # Among feasible, pick the one with lowest assign_cost
                    # (ignoring open cost for now in repair, will account later)
                    costs = assign_cost[i].copy()
                    costs[~feasible] = np.inf
                    f_new = np.argmin(costs)
                    
                    if f_new != f_old:
                        load_rep[f_old] -= d
                        load_rep[f_new] += d
                        repaired_result[i] = f_new
            
            # Check if repair was successful
            if np.any(load_rep > capacities + 1e-9):
                # Repair failed, try a more aggressive approach
                # Or just skip this iteration
                continue
            
            # Evaluate the repaired solution
            score = evaluate(repaired_result)
            
            if score < best_feasible_score:
                best_feasible_score = score
                best_feasible_result = repaired_result.copy()
            
            # Update dual variables using subgradient method
            # Subgradient for lambda_j is (load_j - capacity_j)
            # But we should use the load from the RELAXED solution, not the repaired one
            # because the dual function is based on the relaxed problem.
            subgrad = load - capacities
            
            # Normalize subgradient
            norm_subgrad = np.linalg.norm(subgrad)
            if norm_subgrad < 1e-9:
                break
            
            # Adaptive step size
            # Use a simple diminishing step size
            # Or use the standard subgradient method step size
            # step = alpha * (UB - LB) / ||g||^2
            # We don't have a good LB easily, so use fixed or diminishing step
            
            # Diminishing step size
            alpha = step_size / (it + 1)
            
            # Update lambdas (project onto non-negative orthant since capacities are upper bounds)
            lambdas = lambdas + alpha * subgrad
            lambdas = np.maximum(lambdas, 0)
            
            # Occasionally, try to get a better feasible solution from the current lambda
            # by doing a local search on the repaired solution
            
        return best_feasible_result, best_feasible_score
    
    # Local Search with 1-moves and 2-swaps
    def local_search(result, load, is_open, max_passes=20):
        improved = True
        passes = 0
        
        while improved and passes < max_passes:
            improved = False
            passes += 1
            
            # 1. Try 1-moves (relocation)
            for i in range(n_customers):
                f_old = result[i]
                d = demands[i]
                
                best_delta = 1e-9
                best_f_new = -1
                
                for f_new in range(n_facilities):
                    if f_new == f_old:
                        continue
                    
                    # Check capacity
                    if load[f_new] + d > capacities[f_new]:
                        continue
                    
                    # Calculate delta
                    delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
                    
                    delta_open = 0.0
                    # Check if old facility becomes empty
                    if load[f_old] - d == 0:
                        delta_open -= open_cost[f_old]
                        
                    # Check if new facility was empty
                    if load[f_new] == 0:
                        if not is_open[f_new]:
                            delta_open += open_cost[f_new]
                    
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < best_delta:
                        best_delta = total_delta
                        best_f_new = f_new
                        
                if best_f_new != -1:
                    # Apply move
                    load[f_old] -= d
                    if load[f_old] == 0:
                        is_open[f_old] = False
                        
                    load[best_f_new] += d
                    if load[best_f_new] == d:  # Was 0
                        is_open[best_f_new] = True
                        
                    result[i] = best_f_new
                    improved = True
                    
                    if time.time() > deadline:
                        return result, load, is_open

            # 2. Try 2-swaps (interchange)
            for i in range(n_customers):
                if time.time() > deadline:
                    return result, load, is_open
                for j in range(i + 1, n_customers):
                    f_i = result[i]
                    f_j = result[j]
                    if f_i == f_j:
                        continue
                    
                    d_i = demands[i]
                    d_j = demands[j]
                    
                    # Check capacity feasibility for swap
                    if load[f_j] - d_j + d_i > capacities[f_j]:
                        continue
                    if load[f_i] - d_i + d_j > capacities[f_i]:
                        continue
                    
                    # Calculate assignment cost delta
                    delta_assign = (assign_cost[i, f_j] - assign_cost[i, f_i]) + (assign_cost[j, f_i] - assign_cost[j, f_j])
                    
                    # Calculate open cost delta
                    new_load_i = load[f_i] - d_i + d_j
                    new_load_j = load[f_j] - d_j + d_i
                    
                    old_open_i = is_open[f_i]
                    old_open_j = is_open[f_j]
                    
                    new_open_i = (new_load_i > 0)
                    new_open_j = (new_load_j > 0)
                    
                    delta_open = 0.0
                    if old_open_i and not new_open_i:
                        delta_open -= open_cost[f_i]
                    elif not old_open_i and new_open_i:
                        delta_open += open_cost[f_i]
                        
                    if old_open_j and not new_open_j:
                        delta_open -= open_cost[f_j]
                    elif not old_open_j and new_open_j:
                        delta_open += open_cost[f_j]
                    
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < -1e-9:  # Strict improvement
                        # Apply swap
                        result[i] = f_j
                        result[j] = f_i
                        
                        load[f_i] = new_load_i
                        load[f_j] = new_load_j
                        
                        if new_load_i == 0:
                            is_open[f_i] = False
                        else:
                            is_open[f_i] = True
                            
                        if new_load_j == 0:
                            is_open[f_j] = False
                        else:
                            is_open[f_j] = True
                            
                        improved = True
                        
                        if time.time() > deadline:
                            return result, load, is_open
                            
        return result, load, is_open
    
    # Main loop with restarts
    # 1. Deterministic greedy
    res, load, is_open = greedy_init(randomize=False)
    res, load, is_open = local_search(res, load, is_open)
    score = evaluate(res)
    if score < best_score:
        best_score = score
        best_result = res.copy()
    
    # 2. Lagrangian relaxation
    lr_result, lr_score = lagrangian_relaxation(max_iter=100, step_size=10.0)
    if lr_result is not None and lr_score < best_score:
        # Polish with local search
        load = np.zeros(n_facilities)
        is_open = np.zeros(n_facilities, dtype=bool)
        for i, f in enumerate(lr_result):
            load[f] += demands[i]
            is_open[f] = True
        res, load, is_open = local_search(lr_result, load, is_open)
        score = evaluate(res)
        if score < best_score:
            best_score = score
            best_result = res.copy()
    
    # 3. Randomized restarts with more iterations
    for k in range(200):
        if time.time() > deadline - 5:  # Leave 5s buffer
            break
        
        res, load, is_open = greedy_init(randomize=True, seed=k)
        res, load, is_open = local_search(res, load, is_open)
        score = evaluate(res)
        if score < best_score:
            best_score = score
            best_result = res.copy()
    
    return best_result
