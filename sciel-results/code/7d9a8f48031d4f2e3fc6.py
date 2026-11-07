# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _compute_cost(x, demands, capacities, open_cost, assign_cost):
    """Compute total cost: open costs + assignment costs."""
    n_fac = len(capacities)
    # Check capacity feasibility
    load = np.zeros(n_fac)
    for i in range(len(demands)):
        load[x[i]] += demands[i]
    if np.any(load > capacities):
        return np.inf
    # Open cost
    open_total = 0.0
    for f in range(n_fac):
        if load[f] > 0:
            open_total += open_cost[f]
    # Assignment cost
    assign_total = 0.0
    for i in range(len(demands)):
        assign_total += assign_cost[i, x[i]]
    return open_total + assign_total


def _greedy_init(demands, capacities, assign_cost):
    """Greedy assignment: largest demand first, cheapest feasible facility."""
    n_cust = len(demands)
    n_fac = len(capacities)
    x = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    order = np.argsort(-demands)
    for i in order:
        # Find cheapest feasible facility
        costs = assign_cost[i].copy()
        for f in np.argsort(costs):
            if load[f] + demands[i] <= capacities[f]:
                x[i] = f
                load[f] += demands[i]
                break
        else:
            # Fallback: assign to facility with most remaining capacity
            remaining = capacities - load
            x[i] = np.argmax(remaining)
            load[x[i]] += demands[i]
    return x


def _local_search_1move(x, demands, capacities, open_cost, assign_cost, time_limit=5.0):
    """Local search with 1-move (reassign single customer) improvements."""
    n_cust = len(demands)
    n_fac = len(capacities)
    best_x = x.copy()
    best_cost = _compute_cost(best_x, demands, capacities, open_cost, assign_cost)
    if not np.isfinite(best_cost):
        return best_x, best_cost
    
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[best_x[i]] += demands[i]
    
    start_time = time.time()
    improved = True
    while improved:
        improved = False
        for i in range(n_cust):
            if time.time() - start_time > time_limit:
                break
            current_f = best_x[i]
            current_cost_i = assign_cost[i, current_f]
            # Try moving customer i to each other facility
            for f in range(n_fac):
                if f == current_f:
                    continue
                if load[f] + demands[i] > capacities[f]:
                    continue
                # Compute delta cost
                delta_assign = assign_cost[i, f] - current_cost_i
                # Open cost change: if current_f becomes empty, save open_cost[current_f]
                # if f becomes non-empty, add open_cost[f]
                delta_open = 0.0
                if load[current_f] == demands[i]:
                    # current_f becomes empty
                    delta_open -= open_cost[current_f]
                if load[f] == 0:
                    # f becomes non-empty
                    delta_open += open_cost[f]
                
                total_delta = delta_assign + delta_open
                if total_delta < -1e-10:
                    # Apply move
                    load[current_f] -= demands[i]
                    load[f] += demands[i]
                    best_x[i] = f
                    best_cost += total_delta
                    improved = True
                    break
    return best_x, best_cost


def _local_search_2swap(x, demands, capacities, open_cost, assign_cost, time_limit=10.0):
    """Local search with 2-swap (swap two customers between facilities)."""
    n_cust = len(demands)
    n_fac = len(capacities)
    best_x = x.copy()
    best_cost = _compute_cost(best_x, demands, capacities, open_cost, assign_cost)
    if not np.isfinite(best_cost):
        return best_x, best_cost
    
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[best_x[i]] += demands[i]
    
    # Group customers by facility
    fac_to_cust = {f: [] for f in range(n_fac)}
    for i in range(n_cust):
        fac_to_cust[best_x[i]].append(i)
    
    start_time = time.time()
    improved = True
    while improved:
        improved = False
        for i in range(n_cust):
            if time.time() - start_time > time_limit:
                break
            fi = best_x[i]
            for j in range(i+1, n_cust):
                fj = best_x[j]
                if fi == fj:
                    continue
                # Try swapping i and j
                # Check capacity: fi loses demands[i], gains demands[j]
                # fj loses demands[j], gains demands[i]
                if load[fi] - demands[i] + demands[j] > capacities[fi]:
                    continue
                if load[fj] - demands[j] + demands[i] > capacities[fj]:
                    continue
                
                # Compute delta cost
                delta_assign = (assign_cost[i, fj] + assign_cost[j, fi]) - (assign_cost[i, fi] + assign_cost[j, fj])
                
                # Open cost change
                delta_open = 0.0
                # fi: was load[fi], now load[fi] - demands[i] + demands[j]
                # fj: was load[fj], now load[fj] - demands[j] + demands[i]
                old_load_fi = load[fi]
                old_load_fj = load[fj]
                new_load_fi = old_load_fi - demands[i] + demands[j]
                new_load_fj = old_load_fj - demands[j] + demands[i]
                
                if old_load_fi > 0 and new_load_fi == 0:
                    delta_open -= open_cost[fi]
                elif old_load_fi == 0 and new_load_fi > 0:
                    delta_open += open_cost[fi]
                
                if old_load_fj > 0 and new_load_fj == 0:
                    delta_open -= open_cost[fj]
                elif old_load_fj == 0 and new_load_fj > 0:
                    delta_open += open_cost[fj]
                
                total_delta = delta_assign + delta_open
                if total_delta < -1e-10:
                    # Apply swap
                    load[fi] = new_load_fi
                    load[fj] = new_load_fj
                    best_x[i] = fj
                    best_x[j] = fi
                    # Update fac_to_cust
                    fac_to_cust[fi].remove(i)
                    fac_to_cust[fi].append(j)
                    fac_to_cust[fj].remove(j)
                    fac_to_cust[fj].append(i)
                    best_cost += total_delta
                    improved = True
                    break
    return best_x, best_cost


def _lns_destroy_repair(x, demands, capacities, open_cost, assign_cost, time_limit=15.0):
    """Large Neighborhood Search: destroy a subset of customers, reassign optimally."""
    n_cust = len(demands)
    n_fac = len(capacities)
    best_x = x.copy()
    best_cost = _compute_cost(best_x, demands, capacities, open_cost, assign_cost)
    if not np.isfinite(best_cost):
        return best_x, best_cost
    
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[best_x[i]] += demands[i]
    
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < time_limit:
        # Destroy: pick a random subset of customers to reassign
        k = rng.integers(5, 30)
        destroyed = rng.choice(n_cust, size=k, replace=False)
        
        # Save current assignments for destroyed customers
        old_assign = {i: best_x[i] for i in destroyed}
        
        # Remove destroyed customers from load
        for i in destroyed:
            load[best_x[i]] -= demands[i]
        
        # Solve relaxed assignment for destroyed customers using Hungarian algorithm
        # We need to consider capacity constraints, but for a small subset, we can try
        # a greedy approach with the remaining capacity
        
        # First, try to assign each destroyed customer to the cheapest feasible facility
        # using a cost matrix that includes remaining capacity
        success = True
        for i in destroyed:
            # Find feasible facilities
            feasible = []
            for f in range(n_fac):
                if load[f] + demands[i] <= capacities[f]:
                    feasible.append(f)
            if not feasible:
                success = False
                break
            # Assign to cheapest feasible
            best_f = min(feasible, key=lambda f: assign_cost[i, f])
            best_x[i] = best_f
            load[best_f] += demands[i]
        
        if not success:
            # Restore
            for i in destroyed:
                load[old_assign[i]] += demands[i]
                best_x[i] = old_assign[i]
            continue
        
        # Compute new cost
        new_cost = _compute_cost(best_x, demands, capacities, open_cost, assign_cost)
        if new_cost < best_cost:
            best_cost = new_cost
        else:
            # Restore
            for i in destroyed:
                load[best_x[i]] -= demands[i]
                best_x[i] = old_assign[i]
                load[old_assign[i]] += demands[i]
    
    return best_x, best_cost


def assign(demands, capacities, open_cost, assign_cost):
    """Assign each customer to a facility, minimizing total cost."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    demands = np.asarray(demands)
    capacities = np.asarray(capacities)
    open_cost = np.asarray(open_cost)
    assign_cost = np.asarray(assign_cost)
    
    # Initial greedy solution
    x = _greedy_init(demands, capacities, assign_cost)
    best_x = x.copy()
    best_cost = _compute_cost(best_x, demands, capacities, open_cost, assign_cost)
    
    # Local search: 1-move
    if time.time() < deadline - 30:
        x, cost = _local_search_1move(x, demands, capacities, open_cost, assign_cost, time_limit=10.0)
        if cost < best_cost:
            best_x = x
            best_cost = cost
    
    # Local search: 2-swap
    if time.time() < deadline - 20:
        x, cost = _local_search_2swap(best_x, demands, capacities, open_cost, assign_cost, time_limit=10.0)
        if cost < best_cost:
            best_x = x
            best_cost = cost
    
    # LNS
    if time.time() < deadline - 10:
        x, cost = _lns_destroy_repair(best_x, demands, capacities, open_cost, assign_cost, time_limit=15.0)
        if cost < best_cost:
            best_x = x
            best_cost = cost
    
    # Final 1-move polish
    if time.time() < deadline - 5:
        x, cost = _local_search_1move(best_x, demands, capacities, open_cost, assign_cost, time_limit=5.0)
        if cost < best_cost:
            best_x = x
            best_cost = cost
    
    return best_x.astype(int)
# EVOLVE-BLOCK-END
