"""Lagrangian relaxation + subgradient + local search for capacitated facility location."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_cost(result, demands, open_cost, assign_cost):
    """Compute total cost for a given assignment."""
    total_assign = np.sum(assign_cost[np.arange(len(demands)), result])
    open_mask = np.zeros(len(open_cost), dtype=bool)
    for f in result:
        open_mask[f] = True
    total_open = np.sum(open_cost[open_mask])
    return total_assign + total_open


def _greedy_init(demands, capacities, open_cost, assign_cost):
    """Greedy assignment: largest demand first, cheapest feasible facility."""
    n = len(demands)
    m = len(capacities)
    result = np.full(n, -1, dtype=int)
    load = np.zeros(m)
    
    for i in np.argsort(-demands):
        best_f = -1
        best_cost = np.inf
        for f in range(m):
            if load[f] + demands[i] <= capacities[f]:
                cost = assign_cost[i, f]
                if cost < best_cost:
                    best_cost = cost
                    best_f = f
        if best_f >= 0:
            result[i] = best_f
            load[best_f] += demands[i]
    
    # Handle any unassigned customers (shouldn't happen if total capacity >= total demand)
    for i in range(n):
        if result[i] == -1:
            # Assign to facility with most remaining capacity
            rem_cap = capacities - load
            best_f = np.argmax(rem_cap)
            result[i] = best_f
            load[best_f] += demands[i]
    
    return result


def _local_search(result, demands, capacities, open_cost, assign_cost, max_iters=2000, deadline=None):
    """Local search: try moving each customer to a different facility if it improves cost."""
    n = len(demands)
    m = len(capacities)
    
    # Precompute current loads
    load = np.zeros(m)
    for f in result:
        load[f] += 1  # placeholder, will recompute
    
    load = np.zeros(m)
    for i in range(n):
        load[result[i]] += demands[i]
    
    current_cost = _compute_cost(result, demands, open_cost, assign_cost)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        # Random order for diversity
        order = np.random.permutation(n)
        
        for i in order:
            current_f = result[i]
            d = demands[i]
            
            # Find best alternative facility
            # Cost change if we move customer i from current_f to f:
            # - assign_cost[i, current_f] + assign_cost[i, f]
            # + open_cost change: if current_f becomes empty, -open_cost[current_f]
            #   if f was empty and now has customer, +open_cost[f]
            
            # Try all facilities
            best_f = -1
            best_delta = 0.0
            
            for f in range(m):
                if f == current_f:
                    continue
                if load[f] + d > capacities[f]:
                    continue
                
                # Compute delta in cost
                delta = assign_cost[i, f] - assign_cost[i, current_f]
                
                # Open cost changes
                if load[current_f] == d:  # current_f will be empty
                    delta -= open_cost[current_f]
                if load[f] == 0:  # f was empty
                    delta += open_cost[f]
                
                if delta < best_delta - 1e-9:
                    best_delta = delta
                    best_f = f
            
            if best_f >= 0:
                # Make the move
                load[current_f] -= d
                load[best_f] += d
                result[i] = best_f
                current_cost += best_delta
                improved = True
        
        if not improved:
            break
    
    return result, current_cost


def _subgradient(demands, capacities, open_cost, assign_cost, n_restarts=3, max_iter=300, deadline=None):
    """Subgradient optimization with Lagrangian relaxation."""
    n = len(demands)
    m = len(capacities)
    
    best_result = None
    best_val = np.inf
    
    for restart in range(n_restarts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Initialize multipliers
        scale = float(np.median(assign_cost)) if assign_cost.size else 1.0
        lam = np.random.uniform(0.0, scale, size=m)
        
        step_base = scale / (restart + 1)
        best_restart_val = np.inf
        best_restart_result = None
        
        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
            
            # Solve relaxed problem: assign each customer to min (assign_cost[i,f] + lam[f])
            adjusted = assign_cost + lam[np.newaxis, :]
            relaxed_assign = np.argmin(adjusted, axis=1)
            
            # Check capacity violations
            load = np.zeros(m)
            for i in range(n):
                load[relaxed_assign[i]] += demands[i]
            
            violation = load - capacities
            
            # Update multipliers
            step = step_base / (it + 1)
            lam = np.maximum(0, lam + step * violation)
            
            # Project onto feasible region: if any facility is over capacity,
            # we need to reassign some customers. Use a greedy repair.
            if np.any(violation > 0):
                # Greedy repair: for over-capacity facilities, move customers to least expensive alternatives
                repaired = relaxed_assign.copy()
                load = np.zeros(m)
                for i in range(n):
                    load[repaired[i]] += demands[i]
                
                for f in range(m):
                    if load[f] > capacities[f]:
                        # Find customers at f, move those with best alternative
                        customers_at_f = np.where(repaired == f)[0]
                        excess = load[f] - capacities[f]
                        
                        # Sort by how much we'd save by moving them
                        move_costs = []
                        for i in customers_at_f:
                            # Find best alternative
                            best_alt = -1
                            best_cost = np.inf
                            for alt_f in range(m):
                                if alt_f == f:
                                    continue
                                if load[alt_f] + demands[i] <= capacities[alt_f]:
                                    cost = assign_cost[i, alt_f]
                                    if cost < best_cost:
                                        best_cost = cost
                                        best_alt = alt_f
                            if best_alt >= 0:
                                move_costs.append((best_cost - assign_cost[i, f], i, best_alt))
                        
                        move_costs.sort()
                        for delta_cost, i, alt_f in move_costs:
                            if load[f] <= capacities[f]:
                                break
                            load[f] -= demands[i]
                            load[alt_f] += demands[i]
                            repaired[i] = alt_f
                
                relaxed_assign = repaired
            
            # Compute cost of this relaxed solution (it's feasible now)
            val = _compute_cost(relaxed_assign, demands, open_cost, assign_cost)
            
            if val < best_restart_val:
                best_restart_val = val
                best_restart_result = relaxed_assign.copy()
        
        if best_restart_result is not None and best_restart_val < best_val:
            best_val = best_restart_val
            best_result = best_restart_result
    
    return best_result, best_val


def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands)
    capacities = np.asarray(capacities)
    open_cost = np.asarray(open_cost)
    assign_cost = np.asarray(assign_cost)
    
    n = len(demands)
    m = len(capacities)
    
    # Set deadline with safety margin
    start_time = time.time()
    deadline = start_time + 130.0  # 10s safety margin before 140s limit
    
    # Phase 1: Greedy initialization
    result = _greedy_init(demands, capacities, open_cost, assign_cost)
    current_cost = _compute_cost(result, demands, open_cost, assign_cost)
    
    # Phase 2: Subgradient relaxation for a better starting point
    if time.time() < deadline - 30.0:  # Only if we have enough time
        sg_result, sg_val = _subgradient(demands, capacities, open_cost, assign_cost, 
                                         n_restarts=2, max_iter=100, deadline=deadline - 30.0)
        if sg_result is not None and sg_val < current_cost:
            result = sg_result
            current_cost = sg_val
    
    # Phase 3: Local search
    if time.time() < deadline - 5.0:
        result, current_cost = _local_search(result, demands, capacities, open_cost, 
                                             assign_cost, max_iters=5000, deadline=deadline - 5.0)
    
    # Final validity check
    load = np.zeros(m)
    for i in range(n):
        f = result[i]
        if f < 0 or f >= m:
            # Fallback to greedy
            result = _greedy_init(demands, capacities, open_cost, assign_cost)
            break
        load[f] += demands[i]
    
    if np.any(load > capacities):
        # Fallback to greedy
        result = _greedy_init(demands, capacities, open_cost, assign_cost)
    
    return result
# EVOLVE-BLOCK-END
