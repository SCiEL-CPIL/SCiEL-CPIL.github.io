"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Pre-convert to numpy arrays for speed
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    def calculate_cost(assignment):
        # assignment: array of facility indices
        if np.any(assignment < 0):
            return float('inf')
        total_assign_cost = np.sum(assign_cost[np.arange(n_cust), assignment])
        # Open cost: sum of open_cost for unique facilities used
        used_facilities = np.unique(assignment)
        total_open_cost = np.sum(open_cost[used_facilities])
        return total_assign_cost + total_open_cost

    def is_valid(assignment):
        # Check capacity
        if np.any(assignment < 0) or np.any(assignment >= n_fac):
            return False
        loads = np.zeros(n_fac)
        np.add.at(loads, assignment, demands)
        if np.any(loads > capacities + 1e-9):
            return False
        return True

    def greedy_assign(order, mu=None):
        """
        Assign customers in given order to cheapest feasible facility.
        If mu is provided, use it to adjust costs (Lagrangian).
        """
        result = np.full(n_cust, -1, dtype=int)
        load = np.zeros(n_fac)
        
        # If using Lagrangian, we need to be careful. 
        # Standard Lagrangian: min sum c_ij x_ij + sum mu_j (sum_i d_i x_ij - cap_j)
        # This is equivalent to minimizing sum (c_ij + mu_j * d_i) x_ij - sum mu_j cap_j
        # Since -sum mu_j cap_j is constant, we minimize sum (c_ij + mu_j * d_i) x_ij
        # But we must satisfy capacity constraints in the primal? No, Lagrangian relaxation
        # allows violations if we just solve the subproblem, but we want a feasible solution.
        # Usually, we use the Lagrangian dual to guide the search or construct a heuristic solution.
        # Here, we will use it to bias the greedy choice.
        
        if mu is None:
            mu = np.zeros(n_fac)
            
        # Adjusted cost matrix: c_ij + mu_j * d_i
        # Note: This changes the cost landscape.
        
        for i in order:
            # Calculate effective cost for each facility
            eff_costs = assign_cost[i] + mu * demands[i]
            fac_indices = np.argsort(eff_costs)
            
            assigned = False
            for f in fac_indices:
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    result[i] = f
                    load[f] += demands[i]
                    assigned = True
                    break
            
            if not assigned:
                # Fallback: assign to facility with most remaining capacity
                # Sort by remaining capacity descending
                rem_cap = capacities - load
                fac_indices_cap = np.argsort(-rem_cap)
                for f in fac_indices_cap:
                    if load[f] + demands[i] <= capacities[f] + 1e-9:
                        result[i] = f
                        load[f] += demands[i]
                        assigned = True
                        break
                        
            if not assigned:
                return None # Infeasible
                
        return result

    def local_search(initial_assignment, max_time=20.0):
        """
        Perform local search starting from initial_assignment.
        Uses Relocate and Swap moves.
        Returns the best assignment found.
        """
        current_assignment = initial_assignment.copy()
        current_cost = calculate_cost(current_assignment)
        
        # Precompute loads
        load = np.zeros(n_fac)
        np.add.at(load, current_assignment, demands)
        
        # Track customer count per facility for fast open cost delta calculation
        cust_count = np.zeros(n_fac, dtype=int)
        for i in range(n_cust):
            cust_count[current_assignment[i]] += 1

        local_deadline = time.time() + max_time
        improved = True
        iter_count = 0
        max_iters = 10000
        
        while improved and iter_count < max_iters:
            if time.time() > local_deadline:
                break
            improved = False
            iter_count += 1
            
            # 1. Relocate moves
            # To speed up, we can randomize the order of customers or just iterate 0..N-1
            # Iterating 0..N-1 is deterministic and fine.
            for i in range(n_cust):
                if time.time() > local_deadline:
                    break
                f_from = current_assignment[i]
                
                # Optimization: Only consider facilities that are "plausible"
                # e.g., if cost is too high, skip. But let's keep it simple first.
                
                # Try all other facilities
                # We can sort facilities by assign_cost[i, f] to find good moves faster?
                # But we need to check capacity and open cost.
                # Let's just iterate all f. N=100, F=20. 2000 checks per pass.
                
                best_delta = 0.0
                best_f_to = -1
                
                for f_to in range(n_fac):
                    if f_to == f_from:
                        continue
                    # Check capacity
                    if load[f_to] + demands[i] > capacities[f_to] + 1e-9:
                        continue
                    
                    # Calculate cost change
                    delta_assign = assign_cost[i, f_to] - assign_cost[i, f_from]
                    
                    # Open cost change
                    delta_open = 0.0
                    # If f_from becomes empty (cust_count[f_from] == 1)
                    if cust_count[f_from] == 1:
                        delta_open -= open_cost[f_from]
                    # If f_to was empty (cust_count[f_to] == 0)
                    if cust_count[f_to] == 0:
                        delta_open += open_cost[f_to]
                    
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < best_delta:
                        best_delta = total_delta
                        best_f_to = f_to
                
                if best_f_to != -1:
                    # Accept best move
                    current_assignment[i] = best_f_to
                    load[f_from] -= demands[i]
                    load[best_f_to] += demands[i]
                    cust_count[f_from] -= 1
                    cust_count[best_f_to] += 1
                    current_cost += best_delta
                    improved = True
                    # Note: We do NOT break here. We continue to next customer.
                    # This allows multiple moves in one pass.
                    # However, if we break, we ensure we don't use stale state?
                    # State is updated immediately, so it's fine.
                    # But finding the *best* move for each customer in one pass might be better
                    # than first-improvement. Let's stick to best-improvement per customer.
            
            # 2. Swap moves
            # Iterate over pairs of used facilities
            used_facs = [f for f in range(n_fac) if cust_count[f] > 0]
            
            # To avoid O(N^2) explosion, we only do this if we have time
            if time.time() < local_deadline - 1.0:
                for idx1, f1 in enumerate(used_facs):
                    for f2 in used_facs[idx1+1:]:
                        if time.time() > local_deadline:
                            break
                        
                        # Get customers in f1 and f2
                        # Optimization: Precompute lists?
                        custs_f1 = [i for i in range(n_cust) if current_assignment[i] == f1]
                        custs_f2 = [i for i in range(n_cust) if current_assignment[i] == f2]
                        
                        # If lists are large, this is slow.
                        # N=100, max size 100. 100*100 = 10k checks per pair.
                        # 20*20/2 = 190 pairs. 1.9M checks. Might be too slow in Python.
                        # Let's limit this. Only try swaps if lists are small?
                        # Or just try a subset?
                        # Given 120s budget, and local search is capped at 20s,
                        # 1.9M simple ops might take ~0.2s. Acceptable.
                        
                        best_swap_delta = 0.0
                        best_i, best_j = -1, -1
                        
                        for i in custs_f1:
                            for j in custs_f2:
                                # Check capacity after swap
                                # i goes to f2, j goes to f1
                                if load[f2] - demands[j] + demands[i] > capacities[f2] + 1e-9:
                                    continue
                                if load[f1] - demands[i] + demands[j] > capacities[f1] + 1e-9:
                                    continue
                                
                                # Cost change
                                delta_assign = (assign_cost[i, f2] + assign_cost[j, f1]) - (assign_cost[i, f1] + assign_cost[j, f2])
                                
                                # Open cost change for swap between two non-empty facilities is 0
                                # as established.
                                
                                if delta_assign < best_swap_delta:
                                    best_swap_delta = delta_assign
                                    best_i, best_j = i, j
                        
                        if best_i != -1:
                            # Accept best swap
                            current_assignment[best_i] = f2
                            current_assignment[best_j] = f1
                            load[f1] -= demands[best_i]
                            load[f1] += demands[best_j]
                            load[f2] -= demands[best_j]
                            load[f2] += demands[best_i]
                            # cust_count doesn't change for swaps between non-empty facilities
                            current_cost += best_swap_delta
                            improved = True
                            # Break out of loops? 
                            # If we accept a swap, the state changes significantly.
                            # It's safer to restart the swap search or break out.
                            # Let's break out of all swap loops for this iteration.
                            break
                    if improved:
                        break
                if improved:
                    # Restart main loop to re-evaluate relocates
                    continue

        return current_assignment, current_cost

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
    for _ in range(5):
        order3 = rng.permutation(n_cust)
        cand3 = greedy_assign(order3)
        if cand3 is not None and is_valid(cand3):
            candidates.append(cand3)
            
    # Strategy 4: Cheapest first (min assign cost)
    min_costs = np.min(assign_cost, axis=1)
    order4 = np.argsort(min_costs)
    cand4 = greedy_assign(order4)
    if cand4 is not None and is_valid(cand4):
        candidates.append(cand4)
        
    # Strategy 5: Lagrangian-inspired initializations
    # Try a few random mu vectors to bias the greedy assignment
    # mu should be non-negative.
    for _ in range(5):
        # Simple heuristic for mu: proportional to current load/congestion?
        # Or just random small positive values.
        mu = rng.random(n_fac) * 5.0 # Scale factor
        order5 = np.argsort(-demands) # Large demands first is still good
        cand5 = greedy_assign(order5, mu)
        if cand5 is not None and is_valid(cand5):
            candidates.append(cand5)

    if not candidates:
        # Emergency fallback
        return greedy_assign(np.argsort(-demands))

    # Select best candidate
    best_cand = min(candidates, key=calculate_cost)
    best_cost = calculate_cost(best_cand)
    
    # 2. Local Search on the best candidate
    # We have a global deadline. We have used some time.
    elapsed = time.time() - start_time
    remaining = deadline - start_time - elapsed
    
    # Allocate time for local search
    # If we have plenty of time, run a longer local search.
    # If not, run a shorter one.
    ls_time = min(remaining - 5.0, 60.0) # Cap at 60s, leave 5s buffer
    if ls_time < 1.0:
        ls_time = 1.0
        
    best_assignment, best_cost = local_search(best_cand, max_time=ls_time)
    
    # 3. If we still have time, try to improve other candidates?
    # Or restart local search with different parameters?
    # Given the score improvement from just one good LS, this is likely sufficient.
    # But let's add a second LS run on the second best candidate if time permits.
    
    if time.time() < deadline - 10.0:
        # Sort candidates by cost
        candidates.sort(key=calculate_cost)
        # Take second best if it's different
        if len(candidates) > 1:
            second_cand = candidates[1]
            if not np.array_equal(second_cand, best_cand):
                ls_time2 = min(time.time() - start_time, deadline - time.time() - 5.0)
                if ls_time2 > 5.0:
                    alt_assignment, alt_cost = local_search(second_cand, max_time=ls_time2)
                    if alt_cost < best_cost:
                        best_assignment = alt_assignment
                        best_cost = alt_cost
                        
    return best_assignment

# EVOLVE-BLOCK-END
