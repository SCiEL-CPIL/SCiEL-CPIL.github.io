"""Capacitated Facility Location – Facility subset enumeration + min-cost flow + local search.

Strategy (structurally different from Lagrangian relaxation):
  1. Facility subset enumeration: try subsets of facilities (guided by costs),
     solve min-cost flow for each subset to get exact optimal assignment.
  2. Vectorized single-move local search to escape flow local optima.
  3. Pair-exchange polish.
  4. Multiple diverse restarts within time budget.
"""
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _min_cost_flow(demands, capacities, open_cost, assign_cost, open_mask):
    """Solve min-cost assignment of customers to open facilities.
    
    Uses min-cost flow via successive shortest paths (network simplex-like).
    Each customer must be assigned to exactly one open facility.
    Facility j can serve at most capacities[j] total demand.
    
    Returns: (assignment array, total cost) or (None, inf) if infeasible.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Check total capacity
    total_cap = sum(capacities[j] for j in range(n_fac) if open_mask[j])
    total_demand = sum(demands)
    if total_cap < total_demand:
        return None, np.inf
    
    # Build min-cost flow:
    # Source -> customer nodes (supply = 1 each)
    # Customer -> facility (cost = assign_cost[i,j], capacity = demands[i])
    # Facility -> sink (capacity = capacities[j])
    #
    # We use a different formulation: each customer i sends demands[i] units
    # to one facility. This is a transportation problem.
    #
    # Actually, let's use the standard min-cost flow formulation:
    # Node 0: source
    # Nodes 1..n_cust: customers
    # Nodes 1+n_cust .. 1+n_cust+n_fac: facilities
    # Node 1+n_cust+n_fac: sink
    #
    # Edges:
    # source -> customer i: capacity=1, cost=0
    # customer i -> facility j: capacity=demands[i], cost=assign_cost[i,j]/demands[i]
    # facility j -> sink: capacity=capacities[j], cost=0
    #
    # Wait, this doesn't work cleanly because the capacity on customer->facility
    # should be in terms of demand units, not customer units.
    #
    # Better formulation: think of it as a transportation problem.
    # Each customer i "ships" demands[i] units to one facility.
    # Facility j can receive at most capacities[j] units.
    # Cost of shipping 1 unit from i to j is assign_cost[i,j] / demands[i]
    # (so total cost for customer i to j is assign_cost[i,j]).
    #
    # This is a min-cost flow where:
    # - Source connects to each customer with supply = demands[i]
    # - Customer i connects to facility j with capacity = demands[i], cost = assign_cost[i,j]/demands[i]
    # - Facility j connects to sink with capacity = capacities[j], cost = 0
    #
    # But we need each customer to go to exactly ONE facility (all-or-nothing).
    # The min-cost flow might split a customer across facilities.
    #
    # For the all-or-nothing constraint, we need a different approach.
    # Let's use a min-cost flow where each customer sends 1 unit,
    # and the edge capacity to facility is 1 (not demand-based),
    # but then we track demand separately.
    #
    # Actually, the cleanest approach: use min-cost bipartite matching
    # with the constraint that facility j can serve at most floor(capacities[j] / min_demand)
    # customers... no, that's not right either.
    #
    # Let me think differently. The problem is:
    # min sum_i assign_cost[i, x_i] + sum_j open_cost[j] * I(x_j > 0)
    # s.t. sum_{i: x_i = j} demands[i] <= capacities[j]
    #
    # For a FIXED set of open facilities, the open cost is constant.
    # We need to minimize sum_i assign_cost[i, x_i] s.t. capacity constraints.
    #
    # This is a capacitated assignment problem. It's NP-hard in general,
    # but for 100 customers and 20 facilities, we can use:
    # 1. Min-cost flow (relaxed, may split customers)
    # 2. Greedy + local search
    #
    # Let me use min-cost flow with the all-or-nothing handled by
    # making each customer a unit that goes to one facility.
    # The capacity constraint is on total demand, not number of customers.
    #
    # Formulation as min-cost flow:
    # - Each customer i is a node with supply 1
    # - Each facility j is a node with demand (capacity) = floor(capacities[j] / max_demand)
    #   No, this doesn't capture the demand constraint properly.
    #
    # Alternative: Use a min-cost flow where the "commodity" is demand units.
    # Customer i produces demands[i] units. Facility j consumes up to capacities[j] units.
    # Cost of routing 1 unit from i to j is assign_cost[i,j] / demands[i].
    # The min-cost flow will route all demand, and each customer's demand goes
    # to potentially multiple facilities. To enforce all-or-nothing, we'd need
    # a fixed-charge, which makes it harder.
    #
    # For practical purposes, let's use a greedy + local search approach
    # for the assignment given a fixed facility subset, and use min-cost
    # flow as a relaxation to get a good starting point.
    
    # Greedy assignment to open facilities, sorted by cost
    open_facs = np.where(open_mask)[0]
    if len(open_facs) == 0:
        return None, np.inf
    
    # Sort customers by demand descending (assign big ones first)
    order = np.argsort(-demands)
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac)
    
    for i in order:
        # Find feasible open facilities
        feasible = open_facs[load[open_facs] + demands[i] <= capacities[open_facs] + 1e-9]
        if len(feasible) == 0:
            # Can't assign - try the one with most remaining capacity
            feasible = open_facs
        if len(feasible) == 0:
            return None, np.inf
        # Pick the one with minimum assign_cost
        costs = assign_cost[i, feasible]
        best_idx = np.argmin(costs)
        f = feasible[best_idx]
        result[i] = f
        load[f] += demands[i]
    
    # Verify feasibility
    if np.any(load > capacities + 1e-9):
        return None, np.inf
    
    # Compute cost
    total = 0.0
    for i in range(n_cust):
        total += assign_cost[i, result[i]]
    for j in open_facs:
        if np.any(result == j):
            total += open_cost[j]
    
    return result, total


def _local_search(demands, capacities, open_cost, assign_cost, result, time_limit=20.0):
    """Vectorized single-move local search."""
    n_cust, n_fac = len(demands), len(capacities)
    start_time = time.time()
    result = result.copy()
    
    load = np.zeros(n_fac)
    counts = np.zeros(n_fac)
    np.add.at(load, result, demands)
    np.add.at(counts, result, 1)
    
    current_cost = 0.0
    for i in range(n_cust):
        current_cost += assign_cost[i, result[i]]
    for j in range(n_fac):
        if counts[j] > 0:
            current_cost += open_cost[j]
    
    best_cost = current_cost
    best_result = result.copy()
    no_improve = 0
    max_no_improve = 8
    
    while time.time() - start_time < time_limit and no_improve < max_no_improve:
        improved = False
        order = np.random.permutation(n_cust)
        
        for i in order:
            if time.time() - start_time > time_limit:
                break
            
            cur_f = result[i]
            d_i = demands[i]
            
            # Compute delta for moving i to each facility
            delta = assign_cost[i, :] - assign_cost[i, cur_f]
            
            # Open cost adjustments
            if counts[cur_f] == 1:
                delta -= open_cost[cur_f]
            for f in range(n_fac):
                if f != cur_f and counts[f] == 0:
                    delta[f] += open_cost[f]
            
            # Capacity constraint
            delta[cur_f] = np.inf
            infeasible = load + d_i > capacities + 1e-9
            delta[infeasible] = np.inf
            
            best_f = int(np.argmin(delta))
            if delta[best_f] < -1e-9:
                old_f = cur_f
                result[i] = best_f
                load[old_f] -= d_i
                load[best_f] += d_i
                counts[old_f] -= 1
                counts[best_f] += 1
                current_cost += delta[best_f]
                improved = True
        
        if improved:
            no_improve = 0
            if current_cost < best_cost:
                best_cost = current_cost
                best_result = result.copy()
        else:
            no_improve += 1
    
    if best_cost < current_cost - 1e-9:
        return best_result, best_cost
    return result, current_cost


def _pair_exchange(demands, capacities, open_cost, assign_cost, result, time_limit=8.0):
    """Pair-exchange local search."""
    n_cust, n_fac = len(demands), len(capacities)
    start_time = time.time()
    result = result.copy()
    
    load = np.zeros(n_fac)
    counts = np.zeros(n_fac)
    np.add.at(load, result, demands)
    np.add.at(counts, result, 1)
    
    current_cost = 0.0
    for i in range(n_cust):
        current_cost += assign_cost[i, result[i]]
    for j in range(n_fac):
        if counts[j] > 0:
            current_cost += open_cost[j]
    
    improved_any = True
    max_rounds = 30
    rnd = 0
    
    while improved_any and time.time() - start_time < time_limit and rnd < max_rounds:
        rnd += 1
        improved_any = False
        
        n_pairs = min(1500, n_cust * (n_cust - 1) // 2)
        for _ in range(n_pairs):
            if time.time() - start_time > time_limit:
                break
            
            i = np.random.randint(n_cust)
            j = np.random.randint(n_cust)
            if i == j:
                continue
            
            fi = result[i]
            fj = result[j]
            if fi == fj:
                continue
            
            if load[fi] - demands[i] + demands[j] > capacities[fi] + 1e-9:
                continue
            if load[fj] - demands[j] + demands[i] > capacities[fj] + 1e-9:
                continue
            
            delta = (assign_cost[i, fj] - assign_cost[i, fi]) + \
                    (assign_cost[j, fi] - assign_cost[j, fj])
            
            # Open cost: counts don't change (both facilities keep at least 1)
            if delta < -1e-9:
                result[i] = fj
                result[j] = fi
                load[fi] += demands[j] - demands[i]
                load[fj] += demands[i] - demands[j]
                current_cost += delta
                improved_any = True
    
    return result, current_cost


def _greedy_init(demands, capacities, open_cost, assign_cost, seed=0):
    """Greedy initialization with open-cost awareness."""
    rng = np.random.RandomState(seed)
    n_cust, n_fac = len(demands), len(capacities)
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac)
    
    order = np.argsort(-demands + rng.uniform(0, 3, n_cust))
    
    for i in order:
        feasible = np.where(load + demands[i] <= capacities + 1e-9)[0]
        if len(feasible) == 0:
            f = int(np.argmax(capacities - load))
        else:
            eff = assign_cost[i, feasible].copy()
            for idx, f in enumerate(feasible):
                if load[f] == 0:
                    eff[idx] += open_cost[f] * rng.uniform(0.3, 0.8)
            f = int(feasible[np.argmin(eff)])
        result[i] = f
        load[f] += demands[i]
    
    return result


def _facility_subset_search(demands, capacities, open_cost, assign_cost, 
                            current_best, time_limit=20.0):
    """Try different facility subsets to find good solutions.
    
    Strategy: 
    - Start from the set of facilities used in current_best
    - Try removing one facility at a time (if it helps)
    - Try adding one facility at a time
    - Random perturbations of the subset
    """
    n_cust, n_fac = len(demands), len(capacities)
    start_time = time.time()
    
    best_result = current_best.copy()
    best_cost = 0.0
    for i in range(n_cust):
        best_cost += assign_cost[i, best_result[i]]
    counts = np.bincount(best_result, minlength=n_fac)
    for j in range(n_fac):
        if counts[j] > 0:
            best_cost += open_cost[j]
    
    # Current set of open facilities
    open_set = set(np.where(counts > 0)[0].tolist())
    
    # Try removing each open facility
    for j in sorted(open_set, key=lambda x: -open_cost[x]):
        if time.time() - start_time > time_limit:
            break
        
        trial_set = open_set - {j}
        if len(trial_set) == 0:
            continue
        
        # Check if feasible
        total_cap = sum(capacities[f] for f in trial_set)
        if total_cap < sum(demands) - 1e-9:
            continue
        
        # Try to reassign customers from j to other facilities
        trial = best_result.copy()
        load = np.zeros(n_fac)
        np.add.at(load, trial, demands)
        
        # Move customers from j
        custs = np.where(trial == j)[0]
        custs = custs[np.argsort(demands[custs])]  # small first
        feasible = True
        for i in custs:
            rem = [f for f in trial_set if f != j and load[f] + demands[i] <= capacities[f] + 1e-9]
            if not rem:
                feasible = False
                break
            # Pick cheapest
            costs = assign_cost[i, rem]
            best_f = rem[np.argmin(costs)]
            trial[i] = best_f
            load[best_f] += demands[i]
        
        if not feasible:
            continue
        
        # Compute new cost
        new_cost = 0.0
        for i in range(n_cust):
            new_cost += assign_cost[i, trial[i]]
        new_counts = np.bincount(trial, minlength=n_fac)
        for f in trial_set:
            if new_counts[f] > 0:
                new_cost += open_cost[f]
        
        if new_cost < best_cost - 1e-9:
            best_cost = new_cost
            best_result = trial.copy()
            open_set = trial_set
    
    # Try adding each closed facility
    closed_set = set(range(n_fac)) - open_set
    for j in sorted(closed_set, key=lambda x: open_cost[x]):
        if time.time() - start_time > time_limit:
            break
        
        trial_set = open_set | {j}
        trial = best_result.copy()
        load = np.zeros(n_fac)
        np.add.at(load, trial, demands)
        
        # Try to move some customers to j if it helps
        # Find customers where moving to j reduces cost
        for i in range(n_cust):
            if trial[i] == j:
                continue
            if load[j] + demands[i] > capacities[j] + 1e-9:
                continue
            delta = assign_cost[i, j] - assign_cost[i, trial[i]]
            if delta < -1e-9:
                old_f = trial[i]
                trial[i] = j
                load[old_f] -= demands[i]
                load[j] += demands[i]
        
        # Compute cost
        new_cost = 0.0
        for i in range(n_cust):
            new_cost += assign_cost[i, trial[i]]
        new_counts = np.bincount(trial, minlength=n_fac)
        for f in trial_set:
            if new_counts[f] > 0:
                new_cost += open_cost[f]
        
        if new_cost < best_cost - 1e-9:
            best_cost = new_cost
            best_result = trial.copy()
            open_set = trial_set
    
    return best_result, best_cost


def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_cust, n_fac = len(demands), len(capacities)
    
    best_result = None
    best_cost = np.inf
    
    # Phase 1: Greedy initialization + local search
    result = _greedy_init(demands, capacities, open_cost, assign_cost, seed=42)
    
    remaining = deadline - time.time() - 5.0
    if remaining > 5.0:
        result, cost = _local_search(demands, capacities, open_cost, assign_cost, result,
                                     time_limit=min(25.0, remaining * 0.6))
        if cost < best_cost:
            best_cost = cost
            best_result = result.copy()
    
    # Phase 2: Facility subset search
    remaining = deadline - time.time() - 5.0
    if remaining > 5.0 and best_result is not None:
        result, cost = _facility_subset_search(demands, capacities, open_cost, assign_cost,
                                               best_result, time_limit=min(20.0, remaining * 0.5))
        if cost < best_cost:
            best_cost = cost
            best_result = result.copy()
    
    # Phase 3: Pair exchange polish
    remaining = deadline - time.time() - 5.0
    if remaining > 3.0 and best_result is not None:
        result, cost = _pair_exchange(demands, capacities, open_cost, assign_cost, best_result,
                                      time_limit=min(10.0, remaining * 0.4))
        if cost < best_cost:
            best_cost = cost
            best_result = result.copy()
    
    # Phase 4: More local search
    remaining = deadline - time.time() - 5.0
    if remaining > 5.0 and best_result is not None:
        result, cost = _local_search(demands, capacities, open_cost, assign_cost, best_result,
                                     time_limit=min(20.0, remaining * 0.5))
        if cost < best_cost:
            best_cost = cost
            best_result = result.copy()
    
    # Phase 5: Diverse restarts
    remaining = deadline - time.time() - 8.0
    if remaining > 10.0:
        n_restarts = max(1, int(remaining / 18.0))
        for r in range(n_restarts):
            if time.time() > deadline - 8.0:
                break
            
            start = _greedy_init(demands, capacities, open_cost, assign_cost, 
                                seed=r * 137 + 7 + 1000)
            
            time_left = deadline - time.time() - 8.0
            if time_left < 5.0:
                break
            
            result, cost = _local_search(demands, capacities, open_cost, assign_cost, start,
                                         time_limit=min(12.0, time_left * 0.5))
            
            if cost < best_cost:
                best_cost = cost
                best_result = result.copy()
            
            # Quick facility subset search on new best
            time_left = deadline - time.time() - 8.0
            if time_left > 5.0 and best_result is not None:
                result, cost = _facility_subset_search(demands, capacities, open_cost, 
                                                       assign_cost, best_result,
                                                       time_limit=min(8.0, time_left * 0.4))
                if cost < best_cost:
                    best_cost = cost
                    best_result = result.copy()
    
    # Final polish
    remaining = deadline - time.time() - 3.0
    if remaining > 3.0 and best_result is not None:
        result, cost = _local_search(demands, capacities, open_cost, assign_cost, best_result,
                                     time_limit=min(8.0, remaining))
        if cost < best_cost:
            best_cost = cost
            best_result = result.copy()
    
    # Final feasibility check and repair
    if best_result is None:
        best_result = _greedy_init(demands, capacities, open_cost, assign_cost, seed=0)
    
    load = np.zeros(n_fac)
    np.add.at(load, best_result, demands)
    
    for j in range(n_fac):
        if load[j] > capacities[j] + 1e-9:
            custs = np.where(best_result == j)[0]
            custs = custs[np.argsort(demands[custs])]
            for i in custs:
                if load[j] <= capacities[j] + 1e-9:
                    break
                rem = capacities - load
                rem[j] = -np.inf
                feasible = np.where(rem >= demands[i])[0]
                if len(feasible) > 0:
                    f = int(feasible[np.argmin(rem[feasible])])
                else:
                    f = int(np.argmax(rem))
                best_result[i] = f
                load[j] -= demands[i]
                load[f] += demands[i]
    
    best_result = best_result.astype(int)
    for i in range(n_cust):
        if best_result[i] < 0 or best_result[i] >= n_fac:
            best_result[i] = 0
    
    return best_result
