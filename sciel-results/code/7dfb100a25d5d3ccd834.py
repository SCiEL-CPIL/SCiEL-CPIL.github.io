import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities to minimize total cost.
    Uses Lagrangian relaxation with subgradient updates to guide assignment,
    followed by Iterated Local Search (1-moves and 2-swaps) for final optimization.
    """
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Deadline guard: 115 seconds to be safe within the 120s limit
    deadline = time.time() + 115.0
    
    def evaluate(result):
        """
        Calculates the total cost of an assignment.
        Returns inf if infeasible.
        """
        load = np.bincount(result, weights=demands, minlength=n_facilities)
        
        # Check capacity feasibility
        if np.any(load > capacities + 1e-9):
            return float('inf')
        
        # Calculate open cost
        open_mask = load > 1e-9
        open_sum = np.sum(open_cost[open_mask])
        
        # Calculate assignment cost
        idx = np.arange(n_customers)
        assign_sum = np.sum(assign_cost[idx, result])
        
        return open_sum + assign_sum

    def repair_feasibility(result, load):
        """
        Repairs infeasible solution by moving customers from overloaded facilities
        to underloaded ones with minimal cost increase.
        Returns (result, load, is_open)
        """
        result = result.copy()
        load = load.copy()
        is_open = load > 1e-9
        
        # Identify overloaded facilities
        overloads = load - capacities
        overloaded_facilities = np.where(overloads > 1e-9)[0]
        
        for f_over in overloaded_facilities:
            # Get customers in this facility
            customers_in_f = np.where(result == f_over)[0]
            
            # Sort by demand descending? Or just iterate.
            # We need to remove enough demand to make it feasible.
            excess = load[f_over] - capacities[f_over]
            
            # To minimize cost increase, we should move customers to facilities
            # with low marginal cost.
            # We iterate through customers in f_over.
            # For each, we find the best alternative.
            
            # Optimization: Pre-compute costs? No, dynamic load changes.
            
            for c in customers_in_f:
                if excess <= 1e-9:
                    break
                
                d_c = demands[c]
                # Find best alternative facility
                # Must have capacity: load[f] + d_c <= capacities[f]
                # Minimize cost: assign_cost[c, f] + (open_cost[f] if load[f]==0 else 0)
                
                best_f = -1
                best_cost = float('inf')
                
                for f_new in range(n_facilities):
                    if f_new == f_over:
                        continue
                    if load[f_new] + d_c > capacities[f_new] + 1e-9:
                        continue
                    
                    cost_delta = assign_cost[c, f_new]
                    if load[f_new] == 0:
                        cost_delta += open_cost[f_new]
                    
                    if cost_delta < best_cost:
                        best_cost = cost_delta
                        best_f = f_new
                
                if best_f != -1:
                    # Move customer c to best_f
                    load[f_over] -= d_c
                    if load[f_over] == 0:
                        is_open[f_over] = False
                    load[best_f] += d_c
                    if load[best_f] == d_c:
                        is_open[best_f] = True
                    result[c] = best_f
                    excess -= d_c
                else:
                    # Cannot repair this customer? 
                    # This implies total capacity < total demand or stuck.
                    # In valid instances, total capacity >= total demand usually holds or is implied.
                    # If stuck, we might have to accept infeasibility or break.
                    pass
            
        return result, load, is_open

    def local_search(result, load, is_open):
        """
        Local search with 1-moves and 2-swaps.
        """
        current_cost = evaluate(result)
        if current_cost == float('inf'):
            return result, load, is_open, current_cost

        improved = True
        max_passes = 100
        
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # --- Pass 1: 1-Moves (Relocation) ---
            # Randomize order to avoid bias
            c_order = np.random.permutation(n_customers)
            
            for c in c_order:
                if time.time() > deadline - 10.0:
                    return result, load, is_open, current_cost
                
                f_old = result[c]
                d_c = demands[c]
                
                best_delta = 1e-9
                best_f_new = -1
                
                for f_new in range(n_facilities):
                    if f_new == f_old:
                        continue
                    
                    if load[f_new] + d_c > capacities[f_new] + 1e-9:
                        continue
                    
                    delta_assign = assign_cost[c, f_new] - assign_cost[c, f_old]
                    
                    delta_open = 0.0
                    if load[f_old] == d_c:
                        delta_open -= open_cost[f_old]
                    if load[f_new] == 0.0:
                        delta_open += open_cost[f_new]
                    
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < best_delta:
                        best_delta = total_delta
                        best_f_new = f_new
                        
                if best_f_new != -1:
                    load[f_old] -= d_c
                    if load[f_old] == 0.0:
                        is_open[f_old] = False
                    load[best_f_new] += d_c
                    if load[best_f_new] == d_c:
                        is_open[best_f_new] = True
                    result[c] = best_f_new
                    current_cost += best_delta
                    improved = True

            # --- Pass 2: 2-Swaps (Interchange) ---
            c_order = np.random.permutation(n_customers)
            
            for i in range(n_customers):
                if time.time() > deadline - 10.0:
                    return result, load, is_open, current_cost
                
                c1 = c_order[i]
                f1 = result[c1]
                d1 = demands[c1]
                
                for j in range(i + 1, n_customers):
                    c2 = c_order[j]
                    f2 = result[c2]
                    d2 = demands[c2]
                    
                    if f1 == f2:
                        continue
                    
                    new_load_f1 = load[f1] - d1 + d2
                    new_load_f2 = load[f2] - d2 + d1
                    
                    if new_load_f1 > capacities[f1] + 1e-9:
                        continue
                    if new_load_f2 > capacities[f2] + 1e-9:
                        continue
                    
                    delta_assign = (assign_cost[c1, f2] - assign_cost[c1, f1]) + \
                                   (assign_cost[c2, f1] - assign_cost[c2, f2])
                    
                    # Open cost change for swaps is 0 if both facilities remain non-empty
                    # which they do since demands > 0 and we are swapping.
                    # Unless one facility becomes empty?
                    # If load[f1] == d1, then new_load_f1 = d2. If d2>0, it stays open.
                    # If load[f2] == d2, then new_load_f2 = d1. If d1>0, it stays open.
                    # So open cost delta is 0.
                    
                    total_delta = delta_assign
                    
                    if total_delta < -1e-9:
                        result[c1] = f2
                        result[c2] = f1
                        load[f1] = new_load_f1
                        load[f2] = new_load_f2
                        current_cost += total_delta
                        improved = True
                        break # Restart swap search for this i? Or just break inner loop?
                        # Breaking inner loop is standard first-improvement.
        
        return result, load, is_open, current_cost

    def lagrangian_heuristic(seed=0):
        """
        Lagrangian relaxation heuristic.
        """
        rng = np.random.RandomState(seed)
        
        # Initialize lambda
        lambda_vec = np.zeros(n_facilities)
        
        # Step size
        alpha = 1.0
        best_local_result = None
        best_local_score = float('inf')
        
        max_iters = 1000 # Increased iterations, but guarded by time
        
        for t in range(max_iters):
            if time.time() > deadline - 20.0:
                break
            
            # 1. Solve subproblem: argmin of adjusted costs
            # Cost_{if} = assign_cost[i, f] + lambda_f * demands[i]
            adj_costs = assign_cost + np.outer(demands, lambda_vec)
            current_assignment = np.argmin(adj_costs, axis=1)
            
            # 2. Check feasibility
            load = np.bincount(current_assignment, weights=demands, minlength=n_facilities)
            is_feasible = np.all(load <= capacities + 1e-9)
            
            if is_feasible:
                score = evaluate(current_assignment)
                if score < best_local_score:
                    best_local_score = score
                    best_local_result = current_assignment.copy()
            else:
                # Repair to get a feasible solution for evaluation
                # We do this every few iterations to avoid overhead, or if we haven't found a solution
                if t % 5 == 0 or best_local_result is None:
                    rep_result, rep_load, rep_is_open = repair_feasibility(current_assignment, load)
                    if np.all(rep_load <= capacities + 1e-9):
                        score = evaluate(rep_result)
                        if score < best_local_score:
                            best_local_score = score
                            best_local_result = rep_result.copy()
            
            # 3. Update lambda (Subgradient)
            # g_f = Load_f - Capacity_f
            g = load - capacities
            # Update: lambda = max(0, lambda + alpha * g)
            lambda_vec = np.maximum(0, lambda_vec + alpha * g)
            
            # Decay step size
            alpha *= 0.99 # Slower decay to allow more exploration
            
            # Random perturbation to escape local minima in dual space?
            if t % 50 == 0 and t > 0:
                lambda_vec += rng.uniform(-0.05, 0.05, size=n_facilities)
                lambda_vec = np.maximum(0, lambda_vec)

        return best_local_result, best_local_score

    def greedy_init(randomize=False, seed=None):
        """
        Greedy initialization.
        """
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
        
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        
        # Sort customers by demand descending (Largest Demand First)
        order = np.argsort(-demands)
        
        for i in order:
            d = demands[i]
            feasible = load + d <= capacities
            
            if not np.any(feasible):
                f = np.argmax(capacities - load)
                result[i] = f
                load[f] += d
                continue
            
            marginal_costs = assign_cost[i].copy()
            empty_mask = load == 0
            marginal_costs[empty_mask] += open_cost[empty_mask]
            marginal_costs[~feasible] = np.inf
            
            if randomize:
                sorted_indices = np.argsort(marginal_costs)
                feasible_indices = sorted_indices[marginal_costs[sorted_indices] < np.inf]
                if len(feasible_indices) == 0:
                    f = np.argmax(capacities - load)
                else:
                    top_k = feasible_indices[:min(5, len(feasible_indices))]
                    costs = marginal_costs[top_k]
                    weights = 1.0 / (costs + 1.0)
                    weights /= weights.sum()
                    f = top_k[rng.choice(len(top_k), p=weights)]
            else:
                f = np.argmin(marginal_costs)
            
            result[i] = f
            load[f] += d
            
        return result, load

    # Main Loop: Multi-Start
    best_result = None
    best_score = float('inf')
    
    # 1. Run Lagrangian Heuristic with a few seeds
    for k in range(3):
        if time.time() > deadline - 40.0:
            break
        
        lr_result, lr_score = lagrangian_heuristic(seed=k)
        
        if lr_result is not None and lr_score < best_score:
            best_score = lr_score
            best_result = lr_result.copy()
            
            # Polish with local search
            load = np.bincount(best_result, weights=demands, minlength=n_facilities)
            is_open = load > 1e-9
            ls_result, _, _, ls_score = local_search(best_result, load, is_open)
            
            if ls_score < best_score:
                best_score = ls_score
                best_result = ls_result.copy()
                
    # 2. Greedy Init + Local Search
    if time.time() < deadline - 20.0:
        init_result, init_load = greedy_init(randomize=False, seed=0)
        is_open = init_load > 1e-9
        ls_result, _, _, ls_score = local_search(init_result, init_load, is_open)
        
        if ls_score < best_score:
            best_score = ls_score
            best_result = ls_result.copy()
            
    # 3. Iterated Local Search (Perturbation) from best solution
    if best_result is not None and time.time() < deadline - 10.0:
        current_result = best_result.copy()
        current_score = best_score
        current_load = np.bincount(current_result, weights=demands, minlength=n_facilities)
        current_is_open = current_load > 1e-9
        
        while time.time() < deadline - 2.0:
            # Perturb: move 5 random customers to random feasible facilities
            # This might make it infeasible, so we repair.
            perturbed = current_result.copy()
            for _ in range(5):
                c = np.random.randint(n_customers)
                f_old = perturbed[c]
                d_c = demands[c]
                
                # Find a random feasible new facility
                feasible_facs = np.where(current_load + d_c <= capacities + 1e-9)[0]
                # Exclude old facility? Not strictly necessary for perturbation, but good to move.
                feasible_facs = feasible_facs[feasible_facs != f_old]
                
                if len(feasible_facs) > 0:
                    f_new = feasible_facs[np.random.randint(len(feasible_facs))]
                    # Update load for feasibility check (approximate)
                    current_load[f_old] -= d_c
                    current_load[f_new] += d_c
                    perturbed[c] = f_new
                else:
                    # If no feasible move, pick any facility and let repair handle it
                    f_new = np.random.randint(n_facilities)
                    if f_new != f_old:
                         current_load[f_old] -= d_c
                         current_load[f_new] += d_c
                         perturbed[c] = f_new
            
            # Repair if necessary
            load_check = np.bincount(perturbed, weights=demands, minlength=n_facilities)
            if np.any(load_check > capacities + 1e-9):
                perturbed, load_check, is_open_check = repair_feasibility(perturbed, load_check)
            else:
                is_open_check = load_check > 1e-9
            
            # Local Search
            ls_result, _, _, ls_score = local_search(perturbed, load_check, is_open_check)
            
            if ls_score < current_score:
                current_score = ls_score
                current_result = ls_result.copy()
                if current_score < best_score:
                    best_score = current_score
                    best_result = current_result.copy()
            # Else, keep the old current_result for next perturbation (backtracking)
    
    if best_result is None:
        return np.zeros(n_customers, dtype=int)
        
    return best_result.astype(int)
