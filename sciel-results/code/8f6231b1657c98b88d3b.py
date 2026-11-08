"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


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

    # Local Search: Relocate and Swap
    improved = True
    max_iters = 1000
    iter_count = 0
    
    while improved and iter_count < max_iters:
        if time.time() > deadline:
            break
        improved = False
        iter_count += 1
        
        # --- Relocate Moves ---
        for i in range(n_cust):
            if time.time() > deadline:
                break
            f_from = current_assignment[i]
            d_i = demands[i]
            
            for f_to in sorted_fac_by_cost[i]:
                if f_to == f_from:
                    continue
                if load[f_to] + d_i > capacities[f_to]:
                    continue
                
                delta_assign = assign_cost[i, f_to] - assign_cost[i, f_from]
                delta_open = 0.0
                if cust_count[f_from] == 1:
                    delta_open -= open_cost[f_from]
                if cust_count[f_to] == 0:
                    delta_open += open_cost[f_to]
                    
                total_delta = delta_assign + delta_open
                
                if total_delta < -1e-9:
                    current_assignment[i] = f_to
                    load[f_from] -= d_i
                    load[f_to] += d_i
                    cust_count[f_from] -= 1
                    cust_count[f_to] += 1
                    current_cost += total_delta
                    improved = True
                    break
            
        # --- Swap Moves ---
        used_facs = [f for f in range(n_fac) if cust_count[f] > 0]
        
        for idx1, f1 in enumerate(used_facs):
            if time.time() > deadline:
                break
            for f2 in used_facs[idx1+1:]:
                if time.time() > deadline:
                    break
                
                custs_f1 = [i for i in range(n_cust) if current_assignment[i] == f1]
                custs_f2 = [i for i in range(n_cust) if current_assignment[i] == f2]
                
                found_swap = False
                for i in custs_f1:
                    if found_swap:
                        break
                    for j in custs_f2:
                        if load[f2] - demands[j] + demands[i] > capacities[f2]:
                            continue
                        if load[f1] - demands[i] + demands[j] > capacities[f1]:
                            continue
                        
                        delta_assign = (assign_cost[i, f2] + assign_cost[j, f1]) - (assign_cost[i, f1] + assign_cost[j, f2])
                        
                        if delta_assign < -1e-9:
                            current_assignment[i] = f2
                            current_assignment[j] = f1
                            load[f1] -= demands[i]
                            load[f1] += demands[j]
                            load[f2] -= demands[j]
                            load[f2] += demands[i]
                            current_cost += delta_assign
                            improved = True
                            found_swap = True
                            break
                if found_swap:
                    break
            if found_swap:
                break

    # Final check for validity
    if not is_valid(current_assignment):
        return greedy_assign(np.argsort(-demands))

    # Exact Local Search using LP
    # Try to optimize the assignment within the current set of open facilities
    # This handles the "fixed open set" case optimally.
    open_facs = [f for f in range(n_fac) if cust_count[f] > 0]
    
    # Also try adding one facility? No, that's complex. 
    # Just optimize within current open facilities.
    # If we can improve, great.
    
    # LP formulation:
    # Minimize sum c_ij x_ij
    # s.t. sum_j x_ij = 1 for all i
    #      sum_i d_i x_ij <= cap_j for all j
    #      x_ij = 0 if j not in open_facs
    #      x_ij in {0,1}
    
    # Relax to LP: 0 <= x_ij <= 1
    # Since it's an assignment problem with capacities, if the LP solution is integral, we are done.
    # The constraint matrix is totally unimodular? 
    # Assignment constraints (rows sum to 1) and capacity constraints (cols sum <= cap).
    # This is a bipartite b-matching. The LP relaxation is integral.
    # So solving the LP gives the optimal integer solution for this fixed set of open facilities.
    
    n_vars = n_cust * n_fac
    # c vector
    c = assign_cost.flatten()
    
    # A_eq: sum_j x_ij = 1
    A_eq = np.zeros((n_cust, n_vars))
    for i in range(n_cust):
        for j in range(n_fac):
            A_eq[i, i * n_fac + j] = 1
    
    b_eq = np.ones(n_cust)
    
    # A_ub: sum_i d_i x_ij <= cap_j
    A_ub = np.zeros((n_fac, n_vars))
    for i in range(n_cust):
        for j in range(n_fac):
            A_ub[j, i * n_fac + j] = demands[i]
    
    b_ub = capacities
    
    # Bounds
    bounds = []
    for i in range(n_cust):
        for j in range(n_fac):
            if j in open_facs:
                bounds.append((0, 1))
            else:
                bounds.append((0, 0))
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.status == 0:
            # Extract solution
            x = res.x.reshape((n_cust, n_fac))
            # Round to nearest integer (should be 0 or 1)
            x_int = np.round(x).astype(int)
            
            # Verify validity
            if is_valid(x_int.sum(axis=1)):
                new_cost = calculate_cost(x_int.sum(axis=1))
                if new_cost < current_cost - 1e-9:
                    current_assignment = x_int.sum(axis=1)
                    current_cost = new_cost
                    # Update load and cust_count
                    load = np.zeros(n_fac)
                    cust_count = np.zeros(n_fac, dtype=int)
                    for i in range(n_cust):
                        f = current_assignment[i]
                        load[f] += demands[i]
                        cust_count[f] += 1
    except Exception:
        pass

    return current_assignment

# EVOLVE-BLOCK-END
