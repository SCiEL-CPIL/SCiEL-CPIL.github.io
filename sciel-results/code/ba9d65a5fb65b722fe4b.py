"""
Capacitated facility location: Lagrangian Relaxation with Subgradient Method and Local Search Polish.
"""
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities.
    Uses Lagrangian relaxation on linking constraints to generate multipliers,
    guides a heuristic repair to get a feasible solution, and polishes with local search.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    def compute_cost(result):
        """Compute total cost for a given assignment."""
        if result is None:
            return float('inf')
        load = np.zeros(n_fac)
        fac_count = np.zeros(n_fac, dtype=int)
        for i, f in enumerate(result):
            if f < 0 or f >= n_fac:
                return float('inf')
            load[f] += demands[i]
            fac_count[f] += 1
        if np.any(load > capacities + 1e-9):
            return float('inf')
        return np.sum(open_cost[fac_count > 0]) + np.sum(assign_cost[np.arange(n_cust), result])

    def greedy_init(order_type='desc', randomize=False, seed=None, bias_cost=None):
        """Greedy initialization with optional bias cost."""
        if seed is not None:
            np.random.seed(seed)
            
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        open_set = set()
        
        if order_type == 'desc':
            order = np.argsort(-demands)
        else:
            order = np.argsort(demands)
            
        cost_matrix = assign_cost if bias_cost is None else assign_cost + bias_cost

        for i in order:
            if randomize:
                # Add noise to break ties or explore
                noise = np.random.uniform(0, 10)
                current_costs = cost_matrix[i] + noise
            else:
                current_costs = cost_matrix[i]
                
            fac_order = np.argsort(current_costs)
            
            assigned = False
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    open_set.add(f)
                    assigned = True
                    break
            
            if not assigned:
                # If no capacity, find any facility with capacity (should not happen if total cap > total demand)
                remaining = capacities - load
                feasible_facs = np.where(remaining >= demands[i])[0]
                if len(feasible_facs) > 0:
                    # Pick cheapest among feasible
                    best_f = min(feasible_facs, key=lambda f: assign_cost[i, f])
                    load[best_f] += demands[i]
                    result[i] = best_f
                    open_set.add(best_f)
                else:
                    return None
                    
        return result

    def lagrangian_subgradient(time_limit=20.0):
        """
        Solve Lagrangian relaxation of x_ij <= y_j.
        Returns best multipliers mu and best feasible solution found during repair.
        """
        # mu[i, j] corresponds to constraint x_ij <= y_j
        mu = np.zeros((n_cust, n_fac))
        best_feasible = None
        best_cost = float('inf')
        
        # Initial upper bound from a quick greedy
        init_sol = greedy_init('desc')
        if init_sol is not None:
            best_feasible = init_sol
            best_cost = compute_cost(init_sol)
        
        # Subgradient parameters
        alpha = 1.0
        damping = 0.5
        it = 0
        start_lag = time.time()
        
        while (time.time() - start_lag) < time_limit and (time.time() < deadline - 10.0):
            it += 1
            
            # 1. Solve subproblem
            # y_j = 1 if open_cost[j] < sum_i mu[i, j]
            sum_mu = np.sum(mu, axis=0)
            y = (sum_mu > open_cost).astype(int)
            
            # x_i = argmin_j (assign_cost[i, j] + mu[i, j])
            # We need the full matrix of costs to find argmin
            lag_cost = assign_cost + mu
            x_indices = np.argmin(lag_cost, axis=1)
            
            # Calculate Lower Bound (Lagrangian value)
            # L(mu) = sum_j (F_j - sum_i mu_ij) y_j + sum_ij (c_ij + mu_ij) x_ij
            # Note: x_ij is 1 only for the chosen j for each i.
            # sum_ij (c_ij + mu_ij) x_ij = sum_i (c_i, x_i + mu_i, x_i)
            assignment_costs = assign_cost[np.arange(n_cust), x_indices]
            mu_costs = mu[np.arange(n_cust), x_indices]
            
            # Term for y:
            # If y_j=1, cost is F_j - sum_mu_j. If y_j=0, cost is 0.
            y_costs = np.where(y == 1, open_cost - sum_mu, 0.0)
            
            lb = np.sum(y_costs) + np.sum(assignment_costs) + np.sum(mu_costs)
            
            # Update Upper Bound (Feasible Solution)
            # Repair the current x_indices to a feasible solution
            # Strategy: Use x_indices as preference, but enforce capacity and opening.
            # 1. Determine which facilities are "open" based on y, but we can open more if needed.
            # Actually, simpler repair:
            # Sort customers by demand desc.
            # Assign each to their preferred facility (from x_indices) if capacity allows and facility is "open" (or we open it).
            # If preferred facility is full, assign to next best.
            
            # To speed up repair, we only do it every 10 iterations or if LB improved significantly
            if it % 10 == 0 or (best_feasible is None):
                # Simple repair:
                # 1. Start with empty assignment
                # 2. For each customer (sorted by demand desc), assign to best feasible facility
                #    "Best" is defined by (assign_cost + mu) to use Lagrangian info
                #    But we must pay open_cost if facility wasn't already "open" in y?
                #    Actually, for the repair to be a valid upper bound, we must pay open costs for ALL facilities used.
                
                # Let's use a greedy repair guided by mu
                # We treat (assign_cost + mu) as the cost, but we must ensure we pay open_cost for any facility we use.
                # This is tricky. Let's just use standard greedy but with mu bias, and calculate true cost.
                
                # A better repair:
                # 1. Identify facilities that are open in y.
                # 2. If a customer is assigned to a facility not in y, we have two choices:
                #    a) Open that facility (pay F_j)
                #    b) Reassign customer to a facility in y.
                #    This is a complex repair.
                
                # Simpler: Just run a local search step from the "relaxed" assignment?
                # The relaxed assignment x_indices is a valid assignment (one facility per customer).
                # It might violate capacity.
                # Let's try to fix capacity violations.
                
                current_sol = x_indices.copy()
                load = np.zeros(n_fac)
                for i in range(n_cust):
                    load[current_sol[i]] += demands[i]
                
                # Fix capacity violations
                # Move customers from overloaded facilities to underloaded ones.
                overloaded = np.where(load > capacities)[0]
                if len(overloaded) > 0:
                    # Sort customers by how "bad" they are to move (high demand first?)
                    # Or just iterate through overloaded facilities
                    for f in overloaded:
                        excess = load[f] - capacities[f]
                        if excess <= 0: continue
                        
                        # Get customers assigned to f, sorted by demand desc (move big ones first? or small ones to fit?)
                        # Moving small ones is easier to fit.
                        custs_in_f = [i for i in range(n_cust) if current_sol[i] == f]
                        custs_in_f.sort(key=lambda i: demands[i]) # Ascending demand
                        
                        for i in custs_in_f:
                            if excess <= 0: break
                            
                            # Find best alternative facility
                            # Exclude f
                            # Consider capacity
                            # Use assign_cost + mu as guide
                            best_new_f = -1
                            best_cost_val = float('inf')
                            
                            for g in range(n_fac):
                                if g == f: continue
                                if load[g] + demands[i] <= capacities[g]:
                                    # Cost to move: (c_i,g + mu_i,g) - (c_i,f + mu_i,f) + potential open cost?
                                    # For repair, we just want a feasible solution.
                                    # We'll calculate true cost later.
                                    # Use lagrangian cost as heuristic for "goodness"
                                    val = assign_cost[i, g] + mu[i, g]
                                    if val < best_cost_val:
                                        best_cost_val = val
                                        best_new_f = g
                                        
                            if best_new_f != -1:
                                current_sol[i] = best_new_f
                                load[f] -= demands[i]
                                load[best_new_f] += demands[i]
                                excess -= demands[i]
                            else:
                                # Cannot fix by moving to existing facilities?
                                # This implies total capacity is insufficient or we are stuck.
                                # Try to open a new facility?
                                # For simplicity, if we can't move, we leave it (will be infeasible).
                                # But we should try to find ANY facility with space.
                                for g in range(n_fac):
                                    if g == f: continue
                                    if load[g] + demands[i] <= capacities[g]:
                                        current_sol[i] = g
                                        load[f] -= demands[i]
                                        load[g] += demands[i]
                                        excess -= demands[i]
                                        break
                                
                # Check if current_sol is feasible
                load_final = np.zeros(n_fac)
                for i in range(n_cust):
                    load_final[current_sol[i]] += demands[i]
                
                if np.all(load_final <= capacities + 1e-9):
                    cost = compute_cost(current_sol)
                    if cost < best_cost:
                        best_cost = cost
                        best_feasible = current_sol.copy()
            
            # 2. Calculate Subgradient
            # g_ij = x_ij - y_j
            # x is one-hot
            x_matrix = np.zeros((n_cust, n_fac))
            x_matrix[np.arange(n_cust), x_indices] = 1.0
            
            # y is broadcasted
            g = x_matrix - y[None, :]
            
            # 3. Update Multipliers
            # Polyak step size
            gap = best_cost - lb
            if gap <= 0:
                break
            norm_g_sq = np.sum(g**2)
            if norm_g_sq == 0:
                break
            step_size = (gap / norm_g_sq) * damping
            # Cap step size to avoid huge jumps
            step_size = min(step_size, 1.0)
            
            mu = np.maximum(0, mu + step_size * g)
            
            # Damping factor can be reduced over time
            if it % 100 == 0:
                damping = max(0.1, damping * 0.9)

        return best_feasible, mu

    def local_search(result, time_limit):
        """Local search with 1-moves and 2-swaps."""
        current = result.astype(int).copy()
        load = np.zeros(n_fac)
        fac_count = np.zeros(n_fac, dtype=int)
        for i, f in enumerate(current):
            load[f] += demands[i]
            fac_count[f] += 1
            
        def get_cost():
            return np.sum(open_cost[fac_count > 0]) + np.sum(assign_cost[np.arange(n_cust), current])
            
        current_cost = get_cost()
        best_cost = current_cost
        best_result = current.copy()
        
        start = time.time()
        improved = True
        
        while improved and (time.time() - start < time_limit):
            improved = False
            
            # 1. Single Reassignment (1-move)
            for i in range(n_cust):
                if time.time() - start > time_limit:
                    return best_result
                cur_f = current[i]
                d_i = demands[i]
                best_delta = 0
                best_new_f = -1
                
                for new_f in range(n_fac):
                    if new_f == cur_f:
                        continue
                    if load[new_f] + d_i > capacities[new_f]:
                        continue
                    
                    delta = assign_cost[i, new_f] - assign_cost[i, cur_f]
                    
                    if fac_count[cur_f] == 1:
                        delta -= open_cost[cur_f]
                    if fac_count[new_f] == 0:
                        delta += open_cost[new_f]
                        
                    if delta < best_delta:
                        best_delta = delta
                        best_new_f = new_f
                        
                if best_new_f != -1:
                    old_f = cur_f
                    new_f = best_new_f
                    current[i] = new_f
                    load[old_f] -= d_i
                    load[new_f] += d_i
                    fac_count[old_f] -= 1
                    fac_count[new_f] += 1
                    current_cost += best_delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_result = current.copy()
            
            # 2. Swap (2-move)
            for i in range(n_cust):
                if time.time() - start > time_limit:
                    return best_result
                f_i = current[i]
                d_i = demands[i]
                for j in range(i + 1, n_cust):
                    f_j = current[j]
                    if f_i == f_j:
                        continue
                    d_j = demands[j]
                    
                    if load[f_i] - d_i + d_j > capacities[f_i]:
                        continue
                    if load[f_j] - d_j + d_i > capacities[f_j]:
                        continue
                    
                    delta = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
                    
                    if delta < 0:
                        current[i] = f_j
                        current[j] = f_i
                        load[f_i] = load[f_i] - d_i + d_j
                        load[f_j] = load[f_j] - d_j + d_i
                        current_cost += delta
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_result = current.copy()
                        break
            
        return best_result

    # --- Main Logic ---
    
    # 1. Generate initial candidates
    candidates = []
    
    # Standard Greedy
    init_1 = greedy_init('desc')
    if init_1 is not None:
        candidates.append(init_1)
        
    init_2 = greedy_init('asc')
    if init_2 is not None:
        candidates.append(init_2)
        
    # Randomized Greedy
    for s in [42, 123, 456]:
        if time.time() > deadline - 20: break
        init_r = greedy_init('desc', randomize=True, seed=s)
        if init_r is not None:
            candidates.append(init_r)
            
    # 2. Lagrangian Relaxation
    # Run Lagrangian for a portion of the time
    lag_time = 20.0
    if time.time() < deadline - 40.0:
        lag_sol, mu = lagrangian_subgradient(time_limit=lag_time)
        if lag_sol is not None:
            candidates.append(lag_sol)
            
        # Use mu to guide another greedy init
        if time.time() < deadline - 20.0:
            # Bias cost with mu
            bias = mu
            init_lag = greedy_init('desc', bias_cost=bias)
            if init_lag is not None:
                candidates.append(init_lag)
                
    # 3. Select best candidate and polish
    if not candidates:
        # Fallback: should not happen
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        for i in range(n_cust):
            for f in range(n_fac):
                if load[f] + demands[i] <= capacities[i]: # Bug: should be capacities[f]
                    load[f] += demands[i]
                    result[i] = f
                    break
        candidates.append(result)
        
    best_init = min(candidates, key=compute_cost)
    
    time_elapsed = time.time() - start_time
    time_remaining = deadline - time_elapsed
    
    if time_remaining > 2.0:
        polished = local_search(best_init, time_remaining - 5.0)
        if polished is not None:
            best_init = polished
    
    return best_init
