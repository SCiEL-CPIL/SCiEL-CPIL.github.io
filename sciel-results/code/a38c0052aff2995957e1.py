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

    def greedy_assign(order, dual_prices=None):
        result = np.full(n_cust, -1, dtype=int)
        load = np.zeros(n_fac)
        
        # If dual prices are provided, adjust the effective cost
        if dual_prices is not None:
            adjusted_cost = assign_cost + dual_prices[np.newaxis, :]
            # Re-sort based on adjusted cost
            adj_sorted = np.argsort(adjusted_cost, axis=1)
        else:
            adj_sorted = sorted_fac_by_cost
            
        for i in order:
            d_i = demands[i]
            # Try facilities in order of (potentially adjusted) cost
            for f in adj_sorted[i]:
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

    def local_search(initial_assignment, max_iters=500):
        """
        Perform local search starting from initial_assignment.
        Uses Relocate and Swap moves.
        Returns the best assignment found and its cost.
        """
        current_assignment = initial_assignment.copy()
        current_cost = calculate_cost(current_assignment)
        
        # Precompute loads
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[current_assignment[i]] += demands[i]
            
        # Track customer count per facility for fast open cost delta calculation
        cust_count = np.zeros(n_fac, dtype=int)
        for i in range(n_cust):
            cust_count[current_assignment[i]] += 1

        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iters:
            if time.time() > deadline:
                break
            improved = False
            iter_count += 1
            
            # 1. Relocate moves
            for i in range(n_cust):
                if time.time() > deadline:
                    break
                f_from = current_assignment[i]
                # Try all other facilities sorted by cost
                for f_to in sorted_fac_by_cost[i]:
                    if f_to == f_from:
                        continue
                    # Check capacity
                    if load[f_to] + demands[i] > capacities[f_to]:
                        continue
                    
                    # Calculate cost change
                    delta_assign = assign_cost[i, f_to] - assign_cost[i, f_from]
                    
                    # Open cost change
                    delta_open = 0.0
                    # If f_from becomes empty (cust_count[f_from] == 1)
                    if cust_count[f_from] == 1:
                        delta_open -= open_cost[f_from]
                    # If f_to was empty (cust_count[f_to] == 0)
                        delta_open += open_cost[f_to]
                    
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < -1e-9:
                        # Accept move
                        current_assignment[i] = f_to
                        load[f_from] -= demands[i]
                        load[f_to] += demands[i]
                        cust_count[f_from] -= 1
                        cust_count[f_to] += 1
                        current_cost += total_delta
                        improved = True
                        break # Move to next customer

            # 2. Swap moves
            # Iterate over pairs of used facilities
            used_facs = [f for f in range(n_fac) if cust_count[f] > 0]
            
            found_swap = False
            for idx1, f1 in enumerate(used_facs):
                if time.time() > deadline or found_swap:
                    break
                for f2 in used_facs[idx1+1:]:
                    if time.time() > deadline or found_swap:
                        break
                    
                    # Get customers in f1 and f2
                    custs_f1 = [i for i in range(n_cust) if current_assignment[i] == f1]
                    custs_f2 = [i for i in range(n_cust) if current_assignment[i] == f2]
                    
                    for i in custs_f1:
                        if found_swap:
                            break
                        for j in custs_f2:
                            # Check capacity after swap
                            if load[f2] - demands[j] + demands[i] > capacities[f2]:
                                continue
                            if load[f1] - demands[i] + demands[j] > capacities[f1]:
                                continue
                            
                            # Cost change
                            delta_assign = (assign_cost[i, f2] + assign_cost[j, f1]) - (assign_cost[i, f1] + assign_cost[j, f2])
                            
                            # Open cost change is 0 for swaps between non-empty facilities
                            if delta_assign < -1e-9:
                                # Accept swap
                                current_assignment[i] = f2
                                current_assignment[j] = f1
                                load[f1] -= demands[i]
                                load[f1] += demands[j]
                                load[f2] -= demands[j]
                                load[f2] += demands[i]
                                # cust_count stays the same
                                current_cost += delta_assign
                                improved = True
                                found_swap = True
                                break
            if found_swap:
                # Restart the loop to re-evaluate other moves? 
                # Usually local search continues. The 'improved' flag handles it.
                pass

        return current_assignment, current_cost

    def solve_lp_for_open_set(open_facs, current_assignment):
        """
        Solve LP for fixed set of open facilities to optimize assignment.
        Returns (assignment, cost) or (None, None) if infeasible.
        """
        # LP formulation:
        # Minimize sum c_ij x_ij
        # s.t. sum_j x_ij = 1 for all i
        #      sum_i d_i x_ij <= cap_j for all j
        #      x_ij = 0 if j not in open_facs
        #      x_ij in {0,1}
        
        n_vars = n_cust * n_fac
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
                x = res.x.reshape((n_cust, n_fac))
                x_int = np.round(x).astype(int)
                assignment = x_int.sum(axis=1)
                
                # Verify validity
                if is_valid(assignment):
                    cost = calculate_cost(assignment)
                    return assignment, cost
        except Exception:
            pass
        return None, None

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
        
    # Strategy 5: Lagrangian guided greedy (Dual prices)
    # Simple dual price heuristic: start with 0, adjust based on capacity tightness
    dual_prices = np.zeros(n_fac)
    for _ in range(5):
        # Greedy with current duals
        cand5 = greedy_assign(np.argsort(-demands), dual_prices)
        if cand5 is not None and is_valid(cand5):
            candidates.append(cand5)
        
        # Update dual prices: if facility is full, increase price; if empty, decrease
        loads = np.zeros(n_fac)
        for i in range(n_cust):
            loads[cand5[i]] += demands[i]
        
        for f in range(n_fac):
            if loads[f] >= capacities[f]:
                dual_prices[f] += 1.0
            elif loads[f] == 0:
                dual_prices[f] = max(0, dual_prices[f] - 1.0)
        
        if time.time() > deadline:
            break

    if not candidates:
        # Emergency fallback
        return greedy_assign(np.argsort(-demands))

    # Select best candidate
    best_cand = min(candidates, key=calculate_cost)
    current_assignment = best_cand.copy()
    current_cost = calculate_cost(current_assignment)
    
    # Run local search on best candidate
    current_assignment, current_cost = local_search(current_assignment)
    
    # Also run local search on a few other diverse candidates if time permits
    for cand in candidates:
        if time.time() > deadline:
            break
        if np.array_equal(cand, current_assignment):
            continue
        cand_assign, cand_cost = local_search(cand)
        if cand_cost < current_cost:
            current_assignment = cand_assign
            current_cost = cand_cost

    # Final check for validity
    if not is_valid(current_assignment):
        return greedy_assign(np.argsort(-demands))

    # Exact Local Search using LP
    # Try to optimize the assignment within the current set of open facilities
    open_facs = [f for f in range(n_fac) if current_assignment[f] >= 0] 
    # Wait, current_assignment is customer -> facility.
    # open_facs should be unique facilities used.
    open_facs = list(np.unique(current_assignment))
    
    # Solve LP for this open set
    if time.time() < deadline:
        lp_assign, lp_cost = solve_lp_for_open_set(open_facs, current_assignment)
        if lp_assign is not None and lp_cost < current_cost:
            current_assignment = lp_assign
            current_cost = lp_cost

    # Try adding one facility? 
    # If time permits, try opening one more facility and solving LP.
    if time.time() < deadline - 5.0:
        closed_facs = [f for f in range(n_fac) if f not in open_facs]
        best_new_assign = current_assignment
        best_new_cost = current_cost
        
        for f_new in closed_facs:
            new_open_facs = open_facs + [f_new]
            lp_assign, lp_cost = solve_lp_for_open_set(new_open_facs, current_assignment)
            if lp_assign is not None:
                # Calculate total cost including open cost of new facility
                # The LP cost only includes assignment costs.
                # We need to add open costs for all open facilities.
                # The calculate_cost function does this.
                if lp_cost < best_new_cost:
                    best_new_assign = lp_assign
                    best_new_cost = lp_cost
        
        if best_new_cost < current_cost:
            current_assignment = best_new_assign
            current_cost = best_new_cost

    return current_assignment

# EVOLVE-BLOCK-END
