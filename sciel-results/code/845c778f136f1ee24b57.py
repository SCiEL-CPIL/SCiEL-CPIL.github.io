"""Capacitated Facility Location – stronger solver.

Strategy:
  1. Greedy initialisation (large customers first, cheapest feasible facility).
  2. Intensive local search with two neighbourhoods:
       * single-customer move (with capacity check and open-cost delta)
       * pairwise swap between two facilities
     Both are vectorised where possible for speed.
  3. Time-bounded: returns best feasible solution found before deadline.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _total_cost(demands, capacities, open_cost, assign_cost, result):
    """Compute total objective value of a feasible assignment."""
    n_cust = len(demands)
    n_fac = len(capacities)
    total = 0.0
    for i in range(n_cust):
        total += assign_cost[i, result[i]]
    for j in range(n_fac):
        if np.any(result == j):
            total += open_cost[j]
    return total

def _feasible(demands, capacities, result):
    """Check capacity feasibility."""
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    for i in range(len(demands)):
        load[result[i]] += demands[i]
    return np.all(load <= capacities)

def _greedy_init(demands, capacities, open_cost, assign_cost):
    """Greedy: assign customers from largest demand to smallest,
    choosing the feasible facility with minimum effective cost."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac)
    
    order = np.argsort(-demands)
    for i in order:
        feasible = np.where(load + demands[i] <= capacities)[0]
        if len(feasible) == 0:
            # Fallback: facility with most remaining capacity
            f = int(np.argmax(capacities - load))
        else:
            # Effective cost: assign_cost + proportional open cost if not yet used
            # We track which facilities are "open" implicitly by load > 0
            eff = assign_cost[i, feasible].copy()
            for idx, f in enumerate(feasible):
                if load[f] == 0:
                    eff[idx] += open_cost[f] * 0.5  # heuristic penalty for opening
            f = int(feasible[np.argmin(eff)])
        result[i] = f
        load[f] += demands[i]
    return result

def _local_search(demands, capacities, open_cost, assign_cost, result, time_limit=100.0):
    """Intensive local search with move and swap neighbourhoods."""
    n_cust = len(demands)
    n_fac = len(capacities)
    
    deadline = time.time() + time_limit
    result = result.copy()
    
    # Precompute membership counts
    counts = np.bincount(result, minlength=n_fac)
    load = np.array([sum(demands[i] for i in range(n_cust) if result[i] == j) for j in range(n_fac)], dtype=float)
    
    def compute_cost():
        total = 0.0
        for i in range(n_cust):
            total += assign_cost[i, result[i]]
        for j in range(n_fac):
            if counts[j] > 0:
                total += open_cost[j]
        return total
    
    current_cost = compute_cost()
    no_improve_streak = 0
    max_no_improve = 5  # after this many full passes with no improvement, stop
    
    while time.time() < deadline:
        if no_improve_streak >= max_no_improve:
            break
        improved = False
        
        # --- Single move neighbourhood ---
        for i in range(n_cust):
            if time.time() > deadline:
                break
            cur_f = result[i]
            best_delta = 0.0
            best_f = cur_f
            
            for f in range(n_fac):
                if f == cur_f:
                    continue
                if load[f] + demands[i] > capacities[f]:
                    continue
                delta = assign_cost[i, f] - assign_cost[i, cur_f]
                # Open cost changes
                if counts[f] == 0:
                    delta += open_cost[f]
                if counts[cur_f] == 1:
                    delta -= open_cost[cur_f]
                if delta < best_delta - 1e-9:
                    best_delta = delta
                    best_f = f
            
            if best_f != cur_f:
                # Apply move
                result[i] = best_f
                load[cur_f] -= demands[i]
                load[best_f] += demands[i]
                counts[cur_f] -= 1
                counts[best_f] += 1
                current_cost += best_delta
                improved = True
        
        # --- Swap neighbourhood ---
        for i in range(n_cust):
            if time.time() > deadline:
                break
            for j in range(i + 1, n_cust):
                if time.time() > deadline:
                    break
                fi = result[i]
                fj = result[j]
                if fi == fj:
                    continue
                # Check capacity after swap
                if load[fi] - demands[i] + demands[j] > capacities[fi]:
                    continue
                if load[fj] - demands[j] + demands[i] > capacities[fj]:
                    continue
                delta = (assign_cost[i, fj] - assign_cost[i, fi]) + (assign_cost[j, fi] - assign_cost[j, fj])
                # Open cost: if a facility becomes empty
                if counts[fi] == 1 and counts[fj] > 1:
                    # fi loses i, but gains j -> still has j? No: fi loses i, gains j
                    # After swap: fi has (counts[fi]-1+1)=counts[fi], fj has (counts[fj]-1+1)=counts[fj]
                    # Actually if counts[fi]==1, after removing i and adding j, fi still has 1 (j)
                    # If counts[fj]==1, after removing j and adding i, fj still has 1 (i)
                    # So no open cost change unless one becomes 0
                    pass
                # More careful:
                new_count_fi = counts[fi] - 1 + 1  # loses i, gains j
                new_count_fj = counts[fj] - 1 + 1  # loses j, gains i
                # Open cost delta
                oc_delta = 0.0
                if counts[fi] > 0 and new_count_fi == 0:
                    oc_delta -= open_cost[fi]
                if counts[fj] > 0 and new_count_fj == 0:
                    oc_delta -= open_cost[fj]
                if counts[fi] == 0 and new_count_fi > 0:
                    oc_delta += open_cost[fi]
                if counts[fj] == 0 and new_count_fj > 0:
                    oc_delta += open_cost[fj]
                delta += oc_delta
                
                if delta < -1e-9:
                    # Apply swap
                    result[i] = fj
                    result[j] = fi
                    load[fi] -= demands[i]
                    load[fi] += demands[j]
                    load[fj] -= demands[j]
                    load[fj] += demands[i]
                    counts[fi] = new_count_fi
                    counts[fj] = new_count_fj
                    current_cost += delta
                    improved = True
        
        if improved:
            no_improve_streak = 0
        else:
            no_improve_streak += 1
    
    return result, current_cost

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # 1. Greedy initialisation
    result = _greedy_init(demands, capacities, open_cost, assign_cost)
    
    # 2. Local search
    remaining = deadline - time.time()
    if remaining > 2.0:
        result, _ = _local_search(demands, capacities, open_cost, assign_cost, result, time_limit=max(1.0, remaining - 5.0))
    
    # 3. Final feasibility repair (safety net)
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
    
    for j in range(n_fac):
        if load[j] > capacities[j]:
            # Move excess customers to facilities with space
            custs_at_j = np.where(result == j)[0]
            # Sort by demand ascending to move small ones first
            custs_at_j = custs_at_j[np.argsort(demands[custs_at_j])]
            for i in custs_at_j:
                if load[j] <= capacities[j]:
                    break
                # Find facility with most remaining capacity
                rem = capacities - load
                rem[j] = -np.inf
                f = int(np.argmax(rem))
                if rem[f] >= demands[i]:
                    result[i] = f
                    load[j] -= demands[i]
                    load[f] += demands[i]
    
    # Ensure integer array with valid indices
    result = result.astype(int)
    for i in range(n_cust):
        if result[i] < 0 or result[i] >= n_fac:
            result[i] = 0
    
    return result
# EVOLVE-BLOCK-END
