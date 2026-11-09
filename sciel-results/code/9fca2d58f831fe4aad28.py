# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
import math


def _min_cost_assignment(demands, capacities, assign_cost, allowed):
    """
    Solve capacitated assignment for customers to a fixed set of allowed facilities.
    Returns (assignment array, total_assign_cost) or (None, inf) if infeasible.
    Uses a priority-queue based greedy that assigns customers to the cheapest
    facility with remaining capacity, processing customers in order of largest
    demand first to reduce infeasibility risk.
    """
    n_cust = len(demands)
    n_fac = len(allowed)
    if n_fac == 0:
        return None, float('inf')
    
    # Check total capacity
    if sum(capacities[f] for f in allowed) < sum(demands):
        return None, float('inf')
    
    # Greedy with priority queue: process largest demands first
    order = np.argsort(-demands)
    load = np.zeros(n_fac)
    result = np.full(n_cust, -1)
    
    # Precompute sorted facility indices by cost for each customer
    costs_for_allowed = assign_cost[:, allowed]
    sorted_facs = np.argsort(costs_for_allowed, axis=1)
    
    feasible = True
    for i in order:
        placed = False
        for f_local in sorted_facs[i]:
            f_global = allowed[f_local]
            if load[f_local] + demands[i] <= capacities[f_global]:
                load[f_local] += demands[i]
                result[i] = f_global
                placed = True
                break
        if not placed:
            feasible = False
            break
    
    if not feasible:
        return None, float('inf')
    
    total_cost = sum(assign_cost[i, result[i]] for i in range(n_cust))
    return result, total_cost


def _multi_start_greedy(demands, capacities, assign_cost, allowed, n_trials=10):
    """
    Multi-start greedy: try different customer orderings to find feasible assignments.
    Returns (best_result, best_assign_cost) or (None, inf).
    """
    n_cust = len(demands)
    n_fac = len(allowed)
    
    costs_for_allowed = assign_cost[:, allowed]
    sorted_facs = np.argsort(costs_for_allowed, axis=1)
    
    best_result = None
    best_cost = float('inf')
    
    # First try: largest demand first (most likely to be feasible)
    orders = [np.argsort(-demands)]
    # Then random permutations
    for _ in range(n_trials - 1):
        orders.append(np.random.permutation(n_cust))
    
    for order in orders:
        load = np.zeros(n_fac)
        result = np.full(n_cust, -1)
        feasible = True
        
        for i in order:
            placed = False
            for f_local in sorted_facs[i]:
                f_global = allowed[f_local]
                if load[f_local] + demands[i] <= capacities[f_global]:
                    load[f_local] += demands[i]
                    result[i] = f_global
                    placed = True
                    break
            if not placed:
                feasible = False
                break
        
        if feasible:
            total_cost = sum(assign_cost[i, result[i]] for i in range(n_cust))
            if total_cost < best_cost:
                best_cost = total_cost
                best_result = result.copy()
    
    if best_result is None:
        return None, float('inf')
    
    return best_result, best_cost


def _sa_assignment(demands, capacities, assign_cost, open_cost, allowed, deadline):
    """
    Simulated Annealing for assignment within 'allowed' facilities.
    Returns (best_assignment, best_total_cost).
    """
    n_cust = len(demands)
    n_fac_allowed = len(allowed)
    allowed = np.array(allowed)
    
    # Initialize with a greedy solution
    order = np.argsort(-demands)
    load = np.zeros(n_fac_allowed)
    assignment = np.full(n_cust, -1, dtype=int)
    fac_to_local = {f: idx for idx, f in enumerate(allowed)}
    
    # Greedy fill
    for i in order:
        best_f = -1
        best_cost = float('inf')
        for f_idx in range(n_fac_allowed):
            if load[f_idx] + demands[i] <= capacities[allowed[f_idx]]:
                c = assign_cost[i, allowed[f_idx]]
                if c < best_cost:
                    best_cost = c
                    best_f = f_idx
        if best_f == -1:
            return None, float('inf')
        assignment[i] = allowed[best_f]
        load[best_f] += demands[i]
        
    # Calculate initial cost
    used_mask = np.zeros(n_fac_allowed, dtype=bool)
    for f in assignment:
        used_mask[fac_to_local[f]] = True
    open_cost_total = open_cost[allowed][used_mask].sum()
    assign_cost_total = np.sum([assign_cost[i, assignment[i]] for i in range(n_cust)])
    current_cost = open_cost_total + assign_cost_total
    
    best_assignment = assignment.copy()
    best_cost = current_cost
    
    # SA Parameters
    T = 50.0 # Initial temperature
    cooling_rate = 0.9995
    min_T = 0.5
    
    while T > min_T and time.time() < deadline:
        # Generate move
        move_type = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
        
        delta_cost = 0
        feasible = True
        
        if move_type == 1:
            # 1-move
            i = np.random.randint(0, n_cust)
            f_old_local = fac_to_local[assignment[i]]
            f_new_local = np.random.randint(0, n_fac_allowed)
            if f_new_local == f_old_local:
                continue
            
            # Check capacity
            if load[f_new_local] + demands[i] > capacities[allowed[f_new_local]]:
                continue
            
            # Calculate delta
            delta_assign = assign_cost[i, allowed[f_new_local]] - assign_cost[i, allowed[f_old_local]]
            
            # Open cost change
            new_load_old = load[f_old_local] - demands[i]
            new_load_new = load[f_new_local] + demands[i]
            
            close_old = (load[f_old_local] == demands[i]) # Becomes 0
            open_new = (load[f_new_local] == 0) # Becomes >0
            
            delta_open = 0
            if open_new:
                delta_open += open_cost[allowed[f_new_local]]
            if close_old:
                delta_open -= open_cost[allowed[f_old_local]]
                
            delta_cost = delta_assign + delta_open
            
            # Acceptance
            if delta_cost < 0 or np.random.rand() < math.exp(-delta_cost / T):
                # Update state
                load[f_old_local] = new_load_old
                load[f_new_local] = new_load_new
                assignment[i] = allowed[f_new_local]
                current_cost += delta_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_assignment = assignment.copy()
                    
        elif move_type == 2:
            # 2-swap
            i, j = np.random.choice(n_cust, 2, replace=False)
            f_i_local = fac_to_local[assignment[i]]
            f_j_local = fac_to_local[assignment[j]]
            
            if f_i_local == f_j_local:
                continue
                
            # Check capacity for swap
            new_load_i_at_j = load[f_j_local] - demands[j] + demands[i]
            new_load_j_at_i = load[f_i_local] - demands[i] + demands[j]
            
            if new_load_i_at_j > capacities[allowed[f_j_local]] or new_load_j_at_i > capacities[allowed[f_i_local]]:
                continue
                
            # Calculate delta
            delta_assign = (assign_cost[i, allowed[f_j_local]] + assign_cost[j, allowed[f_i_local]]) - \
                           (assign_cost[i, allowed[f_i_local]] + assign_cost[j, allowed[f_j_local]])
            
            # Delta open = 0 for 2-swaps between distinct facilities
            delta_cost = delta_assign
            
            # Acceptance
            if delta_cost < 0 or np.random.rand() < math.exp(-delta_cost / T):
                # Update
                load[f_i_local] = new_load_j_at_i
                load[f_j_local] = new_load_i_at_j
                assignment[i] = allowed[f_j_local]
                assignment[j] = allowed[f_i_local]
                current_cost += delta_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_assignment = assignment.copy()
                    
        else:
            # 3-cycle
            indices = np.random.choice(n_cust, 3, replace=False)
            i, j, k = indices
            f_i_local = fac_to_local[assignment[i]]
            f_j_local = fac_to_local[assignment[j]]
            f_k_local = fac_to_local[assignment[k]]
            
            # Check distinct facilities
            if len({f_i_local, f_j_local, f_k_local}) < 3:
                continue
                
            # Check capacities
            new_load_i = load[f_i_local] - demands[i] + demands[k]
            new_load_j = load[f_j_local] - demands[j] + demands[i]
            new_load_k = load[f_k_local] - demands[k] + demands[j]
            
            if new_load_i > capacities[allowed[f_i_local]] or \
               new_load_j > capacities[allowed[f_j_local]] or \
               new_load_k > capacities[allowed[f_k_local]]:
                continue
                
            # Delta assign
            delta_assign = (assign_cost[i, allowed[f_j_local]] + 
                            assign_cost[j, allowed[f_k_local]] + 
                            assign_cost[k, allowed[f_i_local]]) - \
                           (assign_cost[i, allowed[f_i_local]] + 
                            assign_cost[j, allowed[f_j_local]] + 
                            assign_cost[k, allowed[f_k_local]])
            
            # Delta open = 0
            delta_cost = delta_assign
            
            if delta_cost < 0 or np.random.rand() < math.exp(-delta_cost / T):
                load[f_i_local] = new_load_i
                load[f_j_local] = new_load_j
                load[f_k_local] = new_load_k
                assignment[i] = allowed[f_j_local]
                assignment[j] = allowed[f_k_local]
                assignment[k] = allowed[f_i_local]
                current_cost += delta_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_assignment = assignment.copy()
                    
        T *= cooling_rate
        
    return best_assignment, best_cost


def assign(demands, capacities, open_cost, assign_cost):
    """
    Capacitated facility location: assign 100 customers to 20 facilities.
    Uses facility subset enumeration + multi-start greedy + local search.
    """
    start_time = time.time()
    deadline = start_time + 135  # Safety margin for 150s limit
    
    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Precompute: for each facility, compute a "value" metric
    avg_assign_cost = assign_cost.mean(axis=0)  # shape (n_fac,)
    
    # Generate candidate subsets of facilities
    candidate_subsets = []
    
    # All facilities
    candidate_subsets.append(set(range(n_fac)))
    
    # Sort by open cost
    by_open = np.argsort(open_cost)
    for k in [5, 8, 10, 12, 15]:
        if k <= n_fac:
            candidate_subsets.append(set(by_open[:k].tolist()))
    
    # Sort by avg assign cost
    by_assign = np.argsort(avg_assign_cost)
    for k in [5, 8, 10, 12, 15]:
        if k <= n_fac:
            candidate_subsets.append(set(by_assign[:k].tolist()))
    
    # Sort by combined metric
    combined = open_cost + avg_assign_cost
    by_combined = np.argsort(combined)
    for k in [5, 8, 10, 12, 15]:
        if k <= n_fac:
            candidate_subsets.append(set(by_combined[:k].tolist()))
    
    # Sort by capacity-adjusted metric: prefer facilities with high capacity and low cost
    cap_per_demand = capacities / (demands.sum() + 1e-9)
    cap_adjusted = open_cost + avg_assign_cost - 10 * cap_per_demand
    by_cap_adj = np.argsort(cap_adjusted)
    for k in [8, 10, 12, 15]:
        if k <= n_fac:
            candidate_subsets.append(set(by_cap_adj[:k].tolist()))
    
    # Remove duplicates
    unique_subsets = []
    seen = set()
    for s in candidate_subsets:
        key = frozenset(s)
        if key not in seen:
            seen.add(key)
            unique_subsets.append(s)
    
    # Evaluate each subset
    best_result = None
    best_total = float('inf')
    
    # Sort subsets by expected quality (fewer facilities with low open cost first)
    def subset_key(s):
        sc = sum(open_cost[f] for f in s)
        sa = sum(avg_assign_cost[f] for f in s)
        return sc + sa
    
    unique_subsets.sort(key=subset_key)
    
    for subset in unique_subsets:
        if time.time() > deadline - 20:
            break
        
        allowed = np.array(sorted(subset))
        result, assign_cost_total = _multi_start_greedy(demands, capacities, assign_cost, allowed, n_trials=5)
        
        if result is None:
            continue
        
        # Compute total cost
        used_mask = np.zeros(n_fac, dtype=bool)
        for f in result:
            used_mask[f] = True
        open_cost_total = open_cost[used_mask].sum()
        total = open_cost_total + assign_cost_total
        
        if total < best_total:
            best_total = total
            best_result = result.copy()
    
    # Simulated Annealing on best result
    if best_result is not None:
        # Determine which facilities are actually used
        used_facilities = set(best_result.tolist())
            
        # Run SA with the currently used facilities
        # Note: SA might close some facilities and open others if we allow it, 
        # but our SA is restricted to 'allowed' set. 
        # To allow changing the set of used facilities, we should pass a larger set 
        # or the full set of allowed facilities for this subset.
        # Let's pass the full 'allowed' set for this subset to SA.
        allowed_list = np.array(sorted(used_facilities))
        # Actually, if we restrict to used_facilities, we can't open new ones.
        # But we can close some.
        # Let's allow SA to operate on the full 'allowed' set of the subset to potentially open others?
        # No, the subset was chosen. Let's stick to the subset's allowed facilities.
        # The 'allowed' variable in the loop is the subset.
            
        sa_result, sa_cost = _sa_assignment(demands, capacities, assign_cost, open_cost, allowed, deadline - 5)
        if sa_result is not None and sa_cost < best_total:
            best_result = sa_result
            best_total = sa_cost
                
        # Recompute cost to be sure
        used_facilities = set(best_result.tolist())
        open_cost_total = sum(open_cost[f] for f in used_facilities)
        assign_cost_total = sum(assign_cost[i, best_result[i]] for i in range(n_cust))
        best_total = open_cost_total + assign_cost_total
    
    # Fallback: if no valid result, use greedy on all facilities
    if best_result is None:
        best_result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        for i in np.argsort(-demands):
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    best_result[i] = f
                    break
    
    return best_result.astype(int)
# EVOLVE-BLOCK-END
