"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations


def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Precompute sorted facility indices for each customer (cheapest first)
    sorted_fac_by_cost = np.argsort(assign_cost, axis=1)
    
    def calculate_cost(assignment):
        total_assign_cost = np.sum(assign_cost[np.arange(n_cust), assignment])
        used_facilities = np.unique(assignment)
        total_open_cost = np.sum(open_cost[used_facilities])
        return total_assign_cost + total_open_cost

    def is_valid(assignment):
        loads = np.zeros(n_fac)
        for i in range(n_cust):
            loads[assignment[i]] += demands[i]
        if np.any(loads > capacities):
            return False
        if np.any(assignment < 0) or np.any(assignment >= n_fac):
            return False
        return True

    def greedy_assign(order, rng=None):
        result = np.full(n_cust, -1, dtype=int)
        load = np.zeros(n_fac)
        for i in order:
            d_i = demands[i]
            # Try facilities in order of cost
            for f in sorted_fac_by_cost[i]:
                if load[f] + d_i <= capacities[f]:
                    result[i] = f
                    load[f] += d_i
                    break
            if result[i] == -1:
                # Fallback: pick facility with most remaining capacity
                best_f = -1
                max_rem = -1
                for f in range(n_fac):
                    rem = capacities[f] - load[f]
                    if rem >= d_i and rem > max_rem:
                        max_rem = rem
                        best_f = f
                if best_f == -1:
                    return None
                result[i] = best_f
                load[best_f] += d_i
        return result

    # 1. Generate diverse initial solutions
    candidates = []
    
    # Strategy 1: Largest demand first
    order1 = np.argsort(-demands)
    cand1 = greedy_assign(order1)
    if cand1 is not None and is_valid(cand1):
        candidates.append(cand1)
        
    # Strategy 2: Smallest demand first
    order2 = np.argsort(demands)
    cand2 = greedy_assign(order2)
    if cand2 is not None and is_valid(cand2):
        candidates.append(cand2)
        
    # Strategy 3: Random orders
    rng = np.random.default_rng(42)
    for _ in range(10):
        order3 = rng.permutation(n_cust)
        cand3 = greedy_assign(order3)
        if cand3 is not None and is_valid(cand3):
            candidates.append(cand3)
            
    # Strategy 4: Cheapest edge first (sort by min assign cost)
    min_costs = np.min(assign_cost, axis=1)
    order4 = np.argsort(min_costs)
    cand4 = greedy_assign(order4)
    if cand4 is not None and is_valid(cand4):
        candidates.append(cand4)
        
    # Strategy 5: Most constrained first (fewest feasible facilities initially)
    # Approximate: facilities with high open cost are less attractive, but let's just use random for now
    # Let's try sorting by demand / capacity ratio? No, just stick to the above.
    
    if not candidates:
        # Emergency fallback
        return greedy_assign(np.argsort(-demands))

    # Select best candidate
    best_cand = min(candidates, key=calculate_cost)
    current_assignment = best_cand.copy()
    current_cost = calculate_cost(current_assignment)
    
    # Precompute loads
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[current_assignment[i]] += demands[i]
    
    # Track customer counts per facility for open cost logic
    cust_count = np.zeros(n_fac, dtype=int)
    for i in range(n_cust):
        cust_count[current_assignment[i]] += 1

    def get_open_cost(assignment):
        # Fast open cost calculation using counts
        # We maintain cust_count, so we can just sum open_cost where count > 0
        return np.sum(open_cost[cust_count > 0])
        
    # Recalculate current_cost to be sure
    # Actually, let's just use the incremental updates in the loop.
    # current_cost is already calculated.

    improved = True
    max_iters = 1000
    iter_count = 0
    
    # Local Search: Relocate and Swap
    while improved and iter_count < max_iters:
        if time.time() > deadline:
            break
        improved = False
        iter_count += 1
        
        # --- Relocate Moves ---
        # Try moving each customer to another facility
        for i in range(n_cust):
            if time.time() > deadline:
                break
            f_from = current_assignment[i]
            d_i = demands[i]
            
            # Try all other facilities, sorted by cost for this customer
            for f_to in sorted_fac_by_cost[i]:
                if f_to == f_from:
                    continue
                # Check capacity
                if load[f_to] + d_i > capacities[f_to]:
                    continue
                
                # Calculate cost change
                delta_assign = assign_cost[i, f_to] - assign_cost[i, f_from]
                
                delta_open = 0.0
                # If f_from becomes empty
                if cust_count[f_from] == 1:
                    delta_open -= open_cost[f_from]
                # If f_to was empty
                if cust_count[f_to] == 0:
                    delta_open += open_cost[f_to]
                    
                total_delta = delta_assign + delta_open
                
                if total_delta < -1e-9:
                    # Accept move
                    current_assignment[i] = f_to
                    load[f_from] -= d_i
                    load[f_to] += d_i
                    cust_count[f_from] -= 1
                    cust_count[f_to] += 1
                    current_cost += total_delta
                    improved = True
                    break # Move to next customer
            
        # --- Swap Moves ---
        # Try swapping two customers between two facilities
        # To keep it efficient, we only try swaps between facilities that are currently used.
        used_facs = [f for f in range(n_fac) if cust_count[f] > 0]
        
        # Limit swap search to avoid O(N^2) explosion in Python loops
        # We iterate over pairs of facilities
        for idx1, f1 in enumerate(used_facs):
            if time.time() > deadline:
                break
            for f2 in used_facs[idx1+1:]:
                if time.time() > deadline:
                    break
                
                # Get customers in f1 and f2
                # To speed this up, we could maintain lists, but for 100 customers, a scan is okay-ish
                # But doing it every swap iteration is expensive.
                # Let's just do a limited number of swap attempts per facility pair.
                
                custs_f1 = [i for i in range(n_cust) if current_assignment[i] == f1]
                custs_f2 = [i for i in range(n_cust) if current_assignment[i] == f2]
                
                # Try swapping each pair
                found_swap = False
                for i in custs_f1:
                    if found_swap:
                        break
                    for j in custs_f2:
                        # Check capacity after swap
                        # i goes to f2, j goes to f1
                        # load[f2] - demands[j] + demands[i] <= capacities[f2]
                        # load[f1] - demands[i] + demands[j] <= capacities[f1]
                        
                        if load[f2] - demands[j] + demands[i] > capacities[f2]:
                            continue
                        if load[f1] - demands[i] + demands[j] > capacities[f1]:
                            continue
                        
                        # Cost change
                        delta_assign = (assign_cost[i, f2] + assign_cost[j, f1]) - (assign_cost[i, f1] + assign_cost[j, f2])
                        
                        # Open cost change:
                        # Swapping between two non-empty facilities does not change the set of used facilities.
                        # Unless one facility becomes empty?
                        # If f1 had only i, and we move i to f2, and move j to f1.
                        # f1 gets j. So f1 is not empty.
                        # f2 gets i. So f2 is not empty.
                        # So delta_open is always 0 for swaps between two currently used facilities.
                        
                        if delta_assign < -1e-9:
                            # Accept swap
                            current_assignment[i] = f2
                            current_assignment[j] = f1
                            load[f1] -= demands[i]
                            load[f1] += demands[j]
                            load[f2] -= demands[j]
                            load[f2] += demands[i]
                            # cust_count stays the same for f1 and f2
                            # cust_count[f1] was C1, now C1 - 1 + 1 = C1.
                            # cust_count[f2] was C2, now C2 - 1 + 1 = C2.
                            current_cost += delta_assign
                            improved = True
                            found_swap = True
                            break
                if found_swap:
                    break
            if found_swap:
                break

    # Final check for validity (should be valid by construction, but just in case)
    if not is_valid(current_assignment):
        # Fallback to a known valid solution
        # This should rarely happen
        return greedy_assign(np.argsort(-demands))

    return current_assignment

# EVOLVE-BLOCK-END
