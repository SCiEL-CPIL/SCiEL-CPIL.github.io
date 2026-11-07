"""Rewritten program with stronger algorithm for capacitated facility location."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _compute_cost(result, demands, capacities, open_cost, assign_cost):
    """Compute total cost of an assignment."""
    n_cust = len(demands)
    n_fac = len(capacities)
    if len(result) != n_cust:
        return np.inf
    for i in range(n_cust):
        if result[i] < 0 or result[i] >= n_fac:
            return np.inf
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
    for f in range(n_fac):
        if load[f] > capacities[f]:
            return np.inf
    total = 0.0
    for f in range(n_fac):
        if load[f] > 0:
            total += open_cost[f]
    for i in range(n_cust):
        total += assign_cost[i, result[i]]
    return total


def _greedy_init(demands, capacities, open_cost, assign_cost):
    """Largest demand first, cheapest feasible facility."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    order = np.argsort(-demands)
    for i in order:
        assigned = False
        for f in np.argsort(assign_cost[i]):
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                assigned = True
                break
        if not assigned:
            rem = capacities - load
            f = np.argmax(rem)
            load[f] += demands[i]
            result[i] = f
    return result


def _random_init(demands, capacities, open_cost, assign_cost, seed=0):
    """Random feasible initialization."""
    rng = np.random.RandomState(seed)
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    order = np.argsort(-demands)
    for i in order:
        assigned = False
        # Try random facilities sorted by cost with some randomness
        fac_order = np.argsort(assign_cost[i] + rng.random(n_fac) * 10)
        for f in fac_order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                assigned = True
                break
        if not assigned:
            rem = capacities - load
            f = np.argmax(rem)
            load[f] += demands[i]
            result[i] = f
    return result


def _local_search(demands, capacities, open_cost, assign_cost, result, time_limit=110.0):
    """Local search: move customers to reduce cost while respecting capacity."""
    start_time = time.time()
    n_cust = len(demands)
    n_fac = len(capacities)
    result = result.copy()
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
    
    best_cost = _compute_cost(result, demands, capacities, open_cost, assign_cost)
    if np.isinf(best_cost):
        return result
    
    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        for i in range(n_cust):
            if time.time() - start_time > time_limit:
                break
            current_f = result[i]
            
            best_delta = 0
            best_f = current_f
            for f in range(n_fac):
                if f == current_f:
                    continue
                if load[f] + demands[i] > capacities[f]:
                    continue
                
                delta = assign_cost[i, f] - assign_cost[i, current_f]
                
                if load[current_f] == demands[i]:
                    delta -= open_cost[current_f]
                
                if load[f] == 0:
                    delta += open_cost[f]
                
                if delta < best_delta:
                    best_delta = delta
                    best_f = f
            
            if best_delta < -1e-9:
                load[current_f] -= demands[i]
                load[best_f] += demands[i]
                result[i] = best_f
                best_cost += best_delta
                improved = True
    
    return result


def _two_opt_search(demands, capacities, open_cost, assign_cost, result, time_limit=30.0):
    """2-opt: move two customers to (potentially) different facilities simultaneously."""
    start_time = time.time()
    n_cust = len(demands)
    n_fac = len(capacities)
    result = result.copy()
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
    
    best_cost = _compute_cost(result, demands, capacities, open_cost, assign_cost)
    if np.isinf(best_cost):
        return result
    
    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        for i in range(n_cust):
            if time.time() - start_time > time_limit:
                break
            fi = result[i]
            for j in range(i + 1, n_cust):
                if time.time() - start_time > time_limit:
                    break
                fj = result[j]
                if fi == fj:
                    continue
                
                # Try all pairs of target facilities (a, b) for (i, j)
                # where i goes to a and j goes to b
                # Skip if a == fi and b == fj (no change)
                best_delta = 0
                best_a = fi
                best_b = fj
                
                # Precompute: what facilities can accept i?
                # What facilities can accept j?
                # For speed, only try a few promising targets
                # Target for i: current fi and up to 3 cheapest feasible
                # Target for j: current fj and up to 3 cheapest feasible
                
                # Build candidate lists
                cand_i = [fi]
                for f in np.argsort(assign_cost[i]):
                    if f != fi and load[f] + demands[i] <= capacities[f]:
                        cand_i.append(f)
                    if len(cand_i) >= 4:
                        break
                
                cand_j = [fj]
                for f in np.argsort(assign_cost[j]):
                    if f != fj and load[f] + demands[j] <= capacities[f]:
                        cand_j.append(f)
                    if len(cand_j) >= 4:
                        break
                
                for a in cand_i:
                    for b in cand_j:
                        if a == fi and b == fj:
                            continue
                        # Check capacity
                        new_load_i = load[fi] - demands[i]
                        new_load_j = load[fj] - demands[j]
                        
                        # If a == fj, then facility fj gets i instead of losing j
                        if a == fj:
                            new_load_a = new_load_j + demands[i]
                            if new_load_a > capacities[a]:
                                continue
                        else:
                            new_load_a = load[a] + demands[i]
                            if new_load_a > capacities[a]:
                                continue
                        
                        if b == fi:
                            new_load_b = new_load_i + demands[j]
                            if new_load_b > capacities[b]:
                                continue
                        else:
                            new_load_b = load[b] + demands[j]
                            if new_load_b > capacities[b]:
                                continue
                        
                        # Compute delta
                        delta = (assign_cost[i, a] + assign_cost[j, b]) - (assign_cost[i, fi] + assign_cost[j, fj])
                        
                        # Open cost changes
                        # Facility fi: loses i, maybe gains j (if b == fi)
                        if b == fi:
                            # fi stays open (still has j)
                            pass
                        elif load[fi] == demands[i]:
                            # fi becomes empty
                            delta -= open_cost[fi]
                        
                        # Facility fj: loses j, maybe gains i (if a == fj)
                        if a == fj:
                            # fj stays open (still has i)
                            pass
                        elif load[fj] == demands[j]:
                            # fj becomes empty
                            delta -= open_cost[fj]
                        
                        # Facility a: gains i
                        if a != fi and a != fj:
                            if load[a] == 0:
                                delta += open_cost[a]
                        elif a == fj:
                            # fj already open, no change
                            pass
                        elif a == fi:
                            # fi already open, no change
                            pass
                        
                        # Facility b: gains j
                        if b != fi and b != fj:
                            if load[b] == 0:
                                delta += open_cost[b]
                        elif b == fi:
                            pass
                        elif b == fj:
                            pass
                        
                        if delta < best_delta - 1e-9:
                            best_delta = delta
                            best_a = a
                            best_b = b
                
                if best_delta < -1e-9:
                    # Apply the move
                    # Remove i from fi, j from fj
                    load[fi] -= demands[i]
                    load[fj] -= demands[j]
                    # Add i to best_a, j to best_b
                    load[best_a] += demands[i]
                    load[best_b] += demands[j]
                    result[i] = best_a
                    result[j] = best_b
                    best_cost += best_delta
                    improved = True
                    break  # Restart j loop with updated state
    
    return result


def _swap_local_search(demands, capacities, open_cost, assign_cost, result, time_limit=110.0):
    """Local search: swap customers between facilities to reduce cost."""
    start_time = time.time()
    n_cust = len(demands)
    n_fac = len(capacities)
    result = result.copy()
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
    
    best_cost = _compute_cost(result, demands, capacities, open_cost, assign_cost)
    if np.isinf(best_cost):
        return result
    
    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        for i in range(n_cust):
            if time.time() - start_time > time_limit:
                break
            fi = result[i]
            for j in range(i + 1, n_cust):
                if time.time() - start_time > time_limit:
                    break
                fj = result[j]
                if fi == fj:
                    continue
                
                if load[fi] - demands[i] + demands[j] > capacities[fi]:
                    continue
                if load[fj] - demands[j] + demands[i] > capacities[fj]:
                    continue
                
                delta = (assign_cost[i, fj] + assign_cost[j, fi]) - (assign_cost[i, fi] + assign_cost[j, fj])
                
                # Open cost: only changes if a facility becomes empty
                if load[fi] == demands[i]:
                    # fi would become empty, but j is coming in
                    # net: fi still has j, so no open cost change
                    pass
                elif load[fj] == demands[j]:
                    # fj would become empty, but i is coming in
                    # net: fj still has i, so no open cost change
                    pass
                
                # Actually, in a swap, no facility becomes empty unless
                # one facility had exactly one customer. Let's re-check:
                # After swap: fi has (load[fi] - demands[i] + demands[j])
                #              fj has (load[fj] - demands[j] + demands[i])
                # If load[fi] == demands[i], after swap fi has demands[j] > 0, so stays open
                # If load[fj] == demands[j], after swap fj has demands[i] > 0, so stays open
                # So no open cost change in a pure swap!
                
                if delta < -1e-9:
                    load[fi] = load[fi] - demands[i] + demands[j]
                    load[fj] = load[fj] - demands[j] + demands[i]
                    result[i] = fj
                    result[j] = fi
                    best_cost += delta
                    improved = True
                    break  # Restart with updated state
    
    return result


def _reoptimize_facilities(demands, capacities, open_cost, assign_cost, result, time_limit=50.0):
    """Try to reassign customers within each facility using Hungarian algorithm."""
    start_time = time.time()
    n_cust = len(demands)
    n_fac = len(capacities)
    result = result.copy()
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
    
    # Group customers by facility
    fac_customers = {f: [] for f in range(n_fac)}
    for i in range(n_cust):
        fac_customers[result[i]].append(i)
    
    for f in range(n_fac):
        if time.time() - start_time > time_limit:
            break
        custs = fac_customers[f]
        if len(custs) < 2:
            continue
        
        # Try to reassign these customers to potentially different facilities
        # But keep it simple: just try moving one customer at a time
        for i in custs:
            if time.time() - start_time > time_limit:
                break
            current_f = result[i]
            best_delta = 0
            best_f = current_f
            
            for nf in range(n_fac):
                if nf == current_f:
                    continue
                if load[nf] + demands[i] > capacities[nf]:
                    continue
                
                delta = assign_cost[i, nf] - assign_cost[i, current_f]
                if load[current_f] == demands[i]:
                    delta -= open_cost[current_f]
                if load[nf] == 0:
                    delta += open_cost[nf]
                
                if delta < best_delta:
                    best_delta = delta
                    best_f = nf
            
            if best_delta < -1e-9:
                load[current_f] -= demands[i]
                load[best_f] += demands[i]
                result[i] = best_f
                fac_customers[current_f].remove(i)
                if best_f != current_f:
                    fac_customers[best_f].append(i)
    
    return result


def assign(demands, capacities, open_cost, assign_cost):
    """Assign customers to facilities using multiple strategies and local search."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Multiple initializations
    initializations = []
    
    # Greedy init
    init1 = _greedy_init(demands, capacities, open_cost, assign_cost)
    initializations.append(init1)
    
    # Random inits
    for seed in [42, 123, 456]:
        if time.time() > deadline - 100:
            break
        init = _random_init(demands, capacities, open_cost, assign_cost, seed=seed)
        initializations.append(init)
    
    # Evaluate and pick best initialization
    best_init = None
    best_init_cost = np.inf
    for init in initializations:
        cost = _compute_cost(init, demands, capacities, open_cost, assign_cost)
        if cost < best_init_cost:
            best_init_cost = cost
            best_init = init
    
    if best_init is None:
        return _greedy_init(demands, capacities, open_cost, assign_cost)
    
    result = best_init
    
    # Local search phases
    # Phase 1: Move local search
    elapsed = time.time() - start_time
    time_limit_move = max(1.0, min(30.0, deadline - 80 - elapsed))
    result = _local_search(demands, capacities, open_cost, assign_cost, result, time_limit_move)
    
    # Phase 2: Swap local search
    elapsed = time.time() - start_time
    time_limit_swap = max(1.0, min(30.0, deadline - 50 - elapsed))
    result = _swap_local_search(demands, capacities, open_cost, assign_cost, result, time_limit_swap)
    
    # Phase 3: 2-opt search (move two customers to new facilities)
    elapsed = time.time() - start_time
    time_limit_2opt = max(1.0, min(30.0, deadline - 25 - elapsed))
    result = _two_opt_search(demands, capacities, open_cost, assign_cost, result, time_limit_2opt)
    
    # Phase 4: Final move local search polish
    elapsed = time.time() - start_time
    time_limit_move2 = max(1.0, min(15.0, deadline - 10 - elapsed))
    result = _local_search(demands, capacities, open_cost, assign_cost, result, time_limit_move2)
    
    # Final check
    final_cost = _compute_cost(result, demands, capacities, open_cost, assign_cost)
    if np.isinf(final_cost):
        # Fallback to greedy
        result = _greedy_init(demands, capacities, open_cost, assign_cost)
    
    return result
# EVOLVE-BLOCK-END
