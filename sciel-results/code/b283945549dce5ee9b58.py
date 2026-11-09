"""Capacitated facility location solver with local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _total_cost(result, demands, capacities, open_cost, assign_cost):
    """Compute total cost for a given assignment."""
    load = np.zeros(len(capacities))
    for i in range(len(demands)):
        load[result[i]] += demands[i]
    # Check capacity
    if np.any(load > capacities):
        return float('inf')
    open_facs = np.where(load > 0)[0]
    cost = np.sum(open_cost[open_facs])
    cost += np.sum(assign_cost[np.arange(len(demands)), result])
    return cost


def _greedy_construct(demands, capacities, open_cost, assign_cost, open_penalty=0.0, rng=None):
    """Greedy construction: largest demand first, cheapest feasible facility.
    open_penalty: add to open_cost to discourage opening more facilities."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    
    # Sort customers by demand descending
    order = np.argsort(-demands)
    
    # Effective open cost with penalty
    eff_open = open_cost + open_penalty
    
    for i in order:
        # Get facilities sorted by assignment cost for this customer
        fac_order = np.argsort(assign_cost[i])
        placed = False
        for f in fac_order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                placed = True
                break
        if not placed:
            # Fallback: find any feasible facility
            for f in range(n_fac):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    placed = True
                    break
            if not placed:
                # Should not happen if problem is feasible
                result[i] = 0
    
    return result


def _local_search(result, demands, capacities, open_cost, assign_cost, time_limit=10.0, rng=None):
    """Local search with reassignment, swap, and facility closing moves."""
    n_cust = len(demands)
    n_fac = len(capacities)
    
    best_result = result.copy()
    best_cost = _total_cost(best_result, demands, capacities, open_cost, assign_cost)
    if np.isinf(best_cost):
        return best_result
    
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[best_result[i]] += demands[i]
    
    start_time = time.time()
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Move 1: Reassign a customer to a different facility
        for i in range(n_cust):
            current_f = best_result[i]
            # Try all other facilities
            for f in range(n_fac):
                if f == current_f:
                    continue
                # Check capacity
                if load[f] + demands[i] <= capacities[f]:
                    # Compute cost change
                    old_cost = assign_cost[i, current_f]
                    new_cost = assign_cost[i, f]
                    
                    # Open cost change
                    open_change = 0.0
                    if load[current_f] == demands[i]:
                        # Closing current facility
                        open_change -= open_cost[current_f]
                    if load[f] == 0:
                        # Opening new facility
                        open_change += open_cost[f]
                    
                    delta = (new_cost - old_cost) + open_change
                    if delta < -1e-9:
                        # Apply move
                        load[current_f] -= demands[i]
                        load[f] += demands[i]
                        best_result[i] = f
                        best_cost += delta
                        improved = True
                        break
        
        # Move 2: Swap two customers between facilities
        for i in range(n_cust):
            if time.time() - start_time >= time_limit:
                break
            for j in range(i + 1, n_cust):
                fi = best_result[i]
                fj = best_result[j]
                if fi == fj:
                    continue
                # Check if swap is feasible
                if load[fi] - demands[i] + demands[j] <= capacities[fi] and \
                   load[fj] - demands[j] + demands[i] <= capacities[fj]:
                    # Compute cost change
                    old_cost = assign_cost[i, fi] + assign_cost[j, fj]
                    new_cost = assign_cost[i, fj] + assign_cost[j, fi]
                    delta = new_cost - old_cost
                    if delta < -1e-9:
                        # Apply swap
                        load[fi] += demands[j] - demands[i]
                        load[fj] += demands[i] - demands[j]
                        best_result[i], best_result[j] = fj, fi
                        best_cost += delta
                        improved = True
        
        # Move 3: Try to close a facility by moving all its customers
        for f in range(n_fac):
            if time.time() - start_time >= time_limit:
                break
            if load[f] == 0:
                continue
            # Find all customers at facility f
            custs_at_f = [i for i in range(n_cust) if best_result[i] == f]
            if len(custs_at_f) == 0:
                continue
            
            # Try to move all customers to other facilities
            # Greedily try to move each customer to cheapest feasible facility
            temp_load = load.copy()
            temp_result = best_result.copy()
            feasible = True
            total_delta = 0.0
            
            for i in custs_at_f:
                # Find cheapest feasible facility (not f)
                best_f_new = -1
                best_delta = float('inf')
                for f2 in range(n_fac):
                    if f2 == f:
                        continue
                    if temp_load[f2] + demands[i] <= capacities[f2]:
                        delta_move = assign_cost[i, f2] - assign_cost[i, f]
                        # Open cost: if f2 is empty, we open it
                        if temp_load[f2] == 0:
                            delta_move += open_cost[f2]
                        if delta_move < best_delta:
                            best_delta = delta_move
                            best_f_new = f2
                if best_f_new == -1:
                    feasible = False
                    break
                temp_load[f] -= demands[i]
                temp_load[best_f_new] += demands[i]
                temp_result[i] = best_f_new
                total_delta += best_delta
            
            if feasible:
                # We're closing facility f, save open_cost[f]
                total_delta -= open_cost[f]
                if total_delta < -1e-9:
                    # Apply move
                    load = temp_load
                    best_result = temp_result
                    best_cost += total_delta
                    improved = True
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 5:
                break
        else:
            no_improve_count = 0
    
    return best_result


def assign(demands, capacities, open_cost, assign_cost):
    """Assign customers to facilities with local search improvement."""
    demands = np.asarray(demands)
    capacities = np.asarray(capacities)
    open_cost = np.asarray(open_cost)
    assign_cost = np.asarray(assign_cost)
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Try multiple initializations with different open penalties
    best_result = None
    best_cost = float('inf')
    
    open_penalties = [0.0, 10.0, 25.0, 50.0, 75.0, 100.0]
    
    for penalty in open_penalties:
        if time.time() > deadline - 30.0:
            break
        
        result = _greedy_construct(demands, capacities, open_cost, assign_cost, 
                                   open_penalty=penalty)
        
        # Local search
        time_left = deadline - time.time() - 5.0  # 5s safety margin
        if time_left < 1.0:
            # Just evaluate without local search
            cost = _total_cost(result, demands, capacities, open_cost, assign_cost)
            if cost < best_cost:
                best_cost = cost
                best_result = result.copy()
            continue
        
        result = _local_search(result, demands, capacities, open_cost, assign_cost,
                              time_limit=time_left)
        
        cost = _total_cost(result, demands, capacities, open_cost, assign_cost)
        if cost < best_cost:
            best_cost = cost
            best_result = result.copy()
    
    if best_result is None:
        # Fallback: simple greedy
        best_result = _greedy_construct(demands, capacities, open_cost, assign_cost)
    
    return best_result
# EVOLVE-BLOCK-END
