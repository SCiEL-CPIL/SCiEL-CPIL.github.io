"""Rewritten with Lagrangian relaxation + local search for capacitated facility location."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_total_cost(demands, capacities, open_cost, assign_cost, assignment):
    """Compute the total cost of an assignment."""
    n = len(demands)
    m = len(capacities)
    load = np.zeros(m)
    for i in range(n):
        load[assignment[i]] += demands[i]
    # Check capacity
    if np.any(load > capacities):
        return np.inf
    # Open cost: facilities with at least one customer
    open_mask = load > 0
    total = np.sum(open_cost[open_mask])
    # Assignment cost
    total += np.sum(assign_cost[np.arange(n), assignment])
    return total


def _greedy_init(demands, capacities, open_cost, assign_cost, rng):
    """Greedy initialization: largest demand first, cheapest feasible facility."""
    n = len(demands)
    m = len(capacities)
    result = np.full(n, -1, dtype=int)
    load = np.zeros(m)
    
    # Sort by demand descending
    order = np.argsort(-demands)
    for i in order:
        # Try facilities in order of assignment cost
        fac_order = np.argsort(assign_cost[i])
        placed = False
        for f in fac_order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                placed = True
                break
        if not placed:
            # Should not happen if total capacity >= total demand
            # Force place at least-capacity facility
            best_f = np.argmax(capacities - load)
            result[i] = best_f
            load[best_f] += demands[i]
    return result


def _lagrangian_solve(demands, capacities, open_cost, assign_cost, rng, max_iter=300):
    """Lagrangian relaxation with subgradient method."""
    n = len(demands)
    m = len(capacities)
    
    best_sol = None
    best_val = np.inf
    
    scale = float(np.median(assign_cost)) if assign_cost.size else 1.0
    
    for restart in range(3):
        lam = rng.uniform(0.0, scale, size=m)
        step_base = scale / (restart + 1)
        
        for it in range(max_iter):
            # Solve relaxed problem: each customer picks min over f of (assign_cost[i,f] + lam[f])
            # But we also need to account for open cost in a smart way
            # For now, just use the dual variables to guide assignment
            effective_cost = assign_cost + lam[np.newaxis, :]
            relaxed_assign = np.argmin(effective_cost, axis=1)
            
            # Compute load
            load = np.zeros(m)
            for i in range(n):
                load[relaxed_assign[i]] += demands[i]
            
            # Subgradient: capacity violation
            grad = load - capacities
            
            # Check if feasible
            if np.all(load <= capacities + 1e-6):
                # Feasible solution - compute actual cost
                val = _compute_total_cost(demands, capacities, open_cost, assign_cost, relaxed_assign)
                if val < best_val:
                    best_val = val
                    best_sol = relaxed_assign.copy()
            
            # Update multipliers (only for violated constraints)
            step = step_base / (1.0 + it * 0.01)
            lam = np.maximum(0.0, lam + step * grad)
    
    return best_sol, best_val


def _local_search(demands, capacities, open_cost, assign_cost, assignment, max_iters=5000):
    """Local search: try moving or swapping customers to reduce cost."""
    n = len(demands)
    m = len(capacities)
    assignment = assignment.copy()
    load = np.zeros(m)
    for i in range(n):
        load[assignment[i]] += demands[i]
    
    current_cost = _compute_total_cost(demands, capacities, open_cost, assign_cost, assignment)
    if np.isinf(current_cost):
        return assignment
    
    improved = True
    iter_count = 0
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        
        # Try moving each customer to a different facility
        for i in range(n):
            current_f = assignment[i]
            # Try all other facilities
            for f in range(m):
                if f == current_f:
                    continue
                if load[f] + demands[i] > capacities[f]:
                    continue
                # Check if this move reduces cost
                # Remove customer i from current_f
                new_load = load.copy()
                new_load[current_f] -= demands[i]
                new_load[f] += demands[i]
                
                # Compute cost change
                # Assignment cost change
                delta_assign = assign_cost[i, f] - assign_cost[i, current_f]
                
                # Open cost change
                old_open = (new_load[current_f] > 0)
                new_open = (new_load[f] > 0)
                delta_open = 0.0
                if not old_open:
                    delta_open -= open_cost[current_f]
                if not new_open:
                    delta_open += open_cost[f]
                
                total_delta = delta_assign + delta_open
                if total_delta < -1e-9:
                    # Make the move
                    load[current_f] -= demands[i]
                    load[f] += demands[i]
                    assignment[i] = f
                    current_cost += total_delta
                    improved = True
                    break  # Restart the loop after a move
    
    return assignment


def _swap_local_search(demands, capacities, open_cost, assign_cost, assignment, max_iters=3000):
    """Local search: try swapping pairs of customers between facilities."""
    n = len(demands)
    m = len(capacities)
    assignment = assignment.copy()
    load = np.zeros(m)
    for i in range(n):
        load[assignment[i]] += demands[i]
    
    current_cost = _compute_total_cost(demands, capacities, open_cost, assign_cost, assignment)
    if np.isinf(current_cost):
        return assignment
    
    improved = True
    iter_count = 0
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        
        # Try swapping pairs
        for i in range(n):
            for j in range(i + 1, n):
                fi = assignment[i]
                fj = assignment[j]
                if fi == fj:
                    continue
                
                # Check capacity after swap
                # i moves to fj, j moves to fi
                if fi != fj:
                    # After swap: fi loses demands[i], gains demands[j]
                    # fj loses demands[j], gains demands[i]
                    new_load_fi = load[fi] - demands[i] + demands[j]
                    new_load_fj = load[fj] - demands[j] + demands[i]
                    if new_load_fi > capacities[fi] or new_load_fj > capacities[fj]:
                        continue
                    
                    # Compute cost change
                    delta_assign = (assign_cost[i, fj] + assign_cost[j, fi]) - \
                                   (assign_cost[i, fi] + assign_cost[j, fj])
                    
                    # Open cost change
                    old_fi_open = (load[fi] - demands[i] > 0) or (demands[j] > 0 and new_load_fi > 0)
                    # Simpler: check if fi becomes empty or fj becomes empty
                    fi_will_be_empty = (load[fi] - demands[i] + demands[j]) <= 0
                    fj_will_be_empty = (load[fj] - demands[j] + demands[i]) <= 0
                    fi_is_empty_now = (load[fi] - demands[i]) <= 0  # after removing i, before adding j
                    fj_is_empty_now = (load[fj] - demands[j]) <= 0
                    
                    delta_open = 0.0
                    # fi: was open (load[fi] > 0), after swap load is new_load_fi
                    if fi_is_empty_now and not fi_will_be_empty:
                        delta_open += open_cost[fi]
                    elif not fi_is_empty_now and fi_will_be_empty:
                        delta_open -= open_cost[fi]
                    # fj: was open (load[fj] > 0), after swap load is new_load_fj
                    if fj_is_empty_now and not fj_will_be_empty:
                        delta_open += open_cost[fj]
                    elif not fj_is_empty_now and fj_will_be_empty:
                        delta_open -= open_cost[fj]
                    
                    total_delta = delta_assign + delta_open
                    if total_delta < -1e-9:
                        # Make the swap
                        load[fi] = new_load_fi
                        load[fj] = new_load_fj
                        assignment[i] = fj
                        assignment[j] = fi
                        current_cost += total_delta
                        improved = True
                        break
            if improved:
                break
    
    return assignment


def assign(demands, capacities, open_cost, assign_cost):
    """Assign each customer to a facility, minimizing total cost."""
    demands = np.asarray(demands, dtype=int)
    capacities = np.asarray(capacities, dtype=int)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n = len(demands)
    m = len(capacities)
    
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_sol = None
    best_val = np.inf
    
    # Strategy 1: Greedy initialization + local search
    for restart in range(5):
        if time.time() > deadline - 30.0:
            break
        
        if restart == 0:
            # Standard greedy
            sol = _greedy_init(demands, capacities, open_cost, assign_cost, rng)
        else:
            # Perturbed greedy: randomize facility ordering slightly
            sol = _greedy_init(demands, capacities, open_cost, assign_cost, rng)
        
        val = _compute_total_cost(demands, capacities, open_cost, assign_cost, sol)
        if val < best_val:
            best_val = val
            best_sol = sol.copy()
        
        # Local search
        if time.time() > deadline - 20.0:
            break
        sol = _local_search(demands, capacities, open_cost, assign_cost, sol, max_iters=3000)
        val = _compute_total_cost(demands, capacities, open_cost, assign_cost, sol)
        if val < best_val:
            best_val = val
            best_sol = sol.copy()
        
        # Swap local search
        if time.time() > deadline - 15.0:
            break
        sol = _swap_local_search(demands, capacities, open_cost, assign_cost, sol, max_iters=2000)
        val = _compute_total_cost(demands, capacities, open_cost, assign_cost, sol)
        if val < best_val:
            best_val = val
            best_sol = sol.copy()
    
    # Strategy 2: Lagrangian relaxation
    if time.time() > deadline - 40.0:
        if best_sol is not None:
            return best_sol.astype(int)
        # Fallback
        return _greedy_init(demands, capacities, open_cost, assign_cost, rng)
    
    lag_sol, lag_val = _lagrangian_solve(demands, capacities, open_cost, assign_cost, rng, max_iter=200)
    if lag_sol is not None:
        val = _compute_total_cost(demands, capacities, open_cost, assign_cost, lag_sol)
        if val < best_val:
            best_val = val
            best_sol = lag_sol.copy()
        
        # Polish with local search
        if time.time() > deadline - 10.0:
            if best_sol is not None:
                return best_sol.astype(int)
            return _greedy_init(demands, capacities, open_cost, assign_cost, rng)
        
        lag_sol = _local_search(demands, capacities, open_cost, assign_cost, lag_sol, max_iters=2000)
        val = _compute_total_cost(demands, capacities, open_cost, assign_cost, lag_sol)
        if val < best_val:
            best_val = val
            best_sol = lag_sol.copy()
        
        lag_sol = _swap_local_search(demands, capacities, open_cost, assign_cost, lag_sol, max_iters=1000)
        val = _compute_total_cost(demands, capacities, open_cost, assign_cost, lag_sol)
        if val < best_val:
            best_val = val
            best_sol = lag_sol.copy()
    
    if best_sol is None:
        best_sol = _greedy_init(demands, capacities, open_cost, assign_cost, rng)
    
    return best_sol.astype(int)
# EVOLVE-BLOCK-END
