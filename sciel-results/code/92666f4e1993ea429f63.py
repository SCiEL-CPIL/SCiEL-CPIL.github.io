import numpy as np
import time
import random
from scipy.optimize import linprog
from scipy.optimize import milp, LinearConstraint, Bounds

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    demands = np.array(demands, dtype=float)
    capacities = np.array(capacities, dtype=float)
    open_cost = np.array(open_cost, dtype=float)
    assign_cost = np.array(assign_cost, dtype=float)
    
    # --- Helper: Compute cost of a solution ---
    def compute_cost(result):
        load = np.zeros(n_fac)
        fac_count = np.zeros(n_fac, dtype=int)
        for i, f in enumerate(result):
            if f < 0 or f >= n_fac:
                return float('inf')
            load[f] += demands[i]
            fac_count[f] += 1
        if np.any(load > capacities + 1e-9):
            return float('inf')
        return np.sum(open_cost[fac_count > 0]) + np.sum(assign_cost[np.arange(n_cust), result])
    
    # --- Helper: Greedy Initialization ---
    def greedy_init(order_func, randomize=False):
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        order = np.argsort(order_func(demands))
        
        for i in order:
            fac_order = np.argsort(assign_cost[i])
            if randomize:
                perturbed = assign_cost[i] + np.random.uniform(0, 1) * 10
                fac_order = np.argsort(perturbed)
            
            assigned = False
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break
            
            if not assigned:
                # Fallback: find any feasible facility
                remaining = capacities - load
                feasible = np.where(remaining >= demands[i])[0]
                if len(feasible) > 0:
                    # Pick min cost among feasible
                    best_f = feasible[np.argmin(assign_cost[i, feasible])]
                    load[best_f] += demands[i]
                    result[i] = best_f
                else:
                    return None
            
        return result

    # --- Helper: LP Rounding Initialization ---
    def lp_rounding_init(lp_sol):
        if lp_sol is None:
            return None
        x_lp = lp_sol[:n_cust * n_fac].reshape(n_cust, n_fac)
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        
        # Sort by demand descending
        order = np.argsort(-demands)
        
        for i in order:
            # Get facilities sorted by LP value descending
            sorted_facs = np.argsort(-x_lp[i])
            assigned = False
            for f in sorted_facs:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break
            
            if not assigned:
                # Fallback to min cost feasible
                feasible = [f for f in range(n_fac) if load[f] + demands[i] <= capacities[f]]
                if feasible:
                    f_best = min(feasible, key=lambda f: assign_cost[i, f])
                    load[f_best] += demands[i]
                    result[f_best] = f_best # Wait, result[i]
                    result[i] = f_best
                else:
                    return None
        
        return result

    # --- 1. Generate Initial Candidates ---
    candidates = []
    
    # Strategy A: MILP Solve (HiGHS)
    # This provides a strong lower bound and potentially an exact or near-exact solution.
    try:
        if time.time() < deadline - 10:
            # Variables: x[i, j] (n_cust * n_fac) and y[j] (n_fac)
            # x is binary, y is binary.
            c_obj = np.concatenate([assign_cost.flatten(), open_cost])
            
            # Constraints:
            # 1. Each customer assigned to exactly one facility: sum_j x[i,j] = 1
            A_eq = np.zeros((n_cust, n_cust * n_fac + n_fac))
            for i in range(n_cust):
                for j in range(n_fac):
                    A_eq[i, i * n_fac + j] = 1.0
            b_eq = np.ones(n_cust)
            
            # 2. Capacity: sum_i demands[i] * x[i,j] <= capacities[j] * y[j]
            #    => sum_i demands[i] * x[i,j] - capacities[j] * y[j] <= 0
            A_ub = np.zeros((n_fac, n_cust * n_fac + n_fac))
            b_ub = np.zeros(n_fac)
            for j in range(n_fac):
                for i in range(n_cust):
                    A_ub[j, i * n_fac + j] = demands[i]
                A_ub[j, n_cust * n_fac + j] = -capacities[j]
            
            # 3. Facility open if any customer assigned: x[i,j] <= y[j]
            #    => x[i,j] - y[j] <= 0
            A_ub2 = np.zeros((n_cust * n_fac, n_cust * n_fac + n_fac))
            b_ub2 = np.zeros(n_cust * n_fac)
            for i in range(n_cust):
                for j in range(n_fac):
                    idx = i * n_fac + j
                    A_ub2[idx, idx] = 1.0
                    A_ub2[idx, n_cust * n_fac + j] = -1.0
            
            A_all = np.vstack([A_ub, A_ub2])
            b_all = np.concatenate([b_ub, b_ub2])
            
            constraint = LinearConstraint(A_all, -np.inf, b_all)
            # x are binary, y are binary
            integrality = np.ones(n_cust * n_fac + n_fac)
            bounds = Bounds(0, 1)
            
            res = milp(c_obj, constraints=constraint, integrality=integrality, bounds=bounds)
            
            if res.status == 0 and res.x is not None:
                x_milp = res.x[:n_cust * n_fac].reshape(n_cust, n_fac)
                # Round to nearest integer (should be close to 0/1)
                x_milp = (x_milp > 0.5).astype(float)
                # Ensure each customer has exactly one assignment
                for i in range(n_cust):
                    if not np.any(x_milp[i]):
                        # Fallback to min cost feasible
                        feasible = [f for f in range(n_fac) if demands[i] <= capacities[f]]
                        if feasible:
                            f_best = min(feasible, key=lambda f: assign_cost[i, f])
                            x_milp[i, f_best] = 1.0
                    else:
                        # If multiple, pick min cost
                        if np.sum(x_milp[i]) > 1:
                            x_milp[i] = 0
                            f_best = np.argmin(assign_cost[i])
                            x_milp[i, f_best] = 1.0
                
                # Check capacity
                load = np.sum(x_milp * demands[:, np.newaxis], axis=0)
                if np.all(load <= capacities + 1e-9):
                    result = np.argmax(x_milp, axis=1).astype(int)
                    candidates.append(result)
                else:
                    # Use LP rounding logic on the MILP solution? Or just skip.
                    # Let's try rounding with capacity check in a greedy way if direct rounding fails.
                    init_lp = lp_rounding_init(res.x)
                    if init_lp is not None:
                        candidates.append(init_lp)
                        
    except Exception:
        pass
        
    # Strategy B: Greedy (Largest Demand First)
    init_g1 = greedy_init(lambda d: -d)
    if init_g1 is not None:
        candidates.append(init_g1)
        
    # Strategy C: Greedy (Smallest Demand First)
    init_g2 = greedy_init(lambda d: d)
    if init_g2 is not None:
        candidates.append(init_g2)
        
    # Strategy D: Randomized Greedy
    for _ in range(3):
        if time.time() > deadline - 15:
            break
        init_r = greedy_init(lambda d: -d, randomize=True)
        if init_r is not None:
            candidates.append(init_r)
            
    if not candidates:
        # Last resort
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        for i in range(n_cust):
            for f in range(n_fac):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break
        candidates.append(result)

    # Pick best initial candidate
    best_init = min(candidates, key=compute_cost)
    
    # --- 2. Local Search Polish ---
    def _local_search(result, time_limit):
        current = result.astype(int).copy()
        load = np.zeros(n_fac)
        fac_count = np.zeros(n_fac, dtype=int)
        for i, f in enumerate(current):
            load[f] += demands[i]
            fac_count[f] += 1
            
        def get_cost():
            return np.sum(open_cost[fac_count > 0]) + np.sum(assign_cost[np.arange(n_cust), current])
            
        current_cost = get_cost()
        best_cost = current_cost
        best_result = current.copy()
        
        start = time.time()
        improved = True
        
        # Precompute arange for speed
        idx = np.arange(n_cust)
        
        while improved and (time.time() - start < time_limit):
            improved = False
            
            # 1. Single Reassignment (Move)
            # Iterate customers in random order to avoid bias? Or standard order?
            # Standard order is fine, but maybe shuffle occasionally?
            # Let's stick to standard for determinism in local search steps, 
            # but the multiple starts provide diversity.
            
            for i in range(n_cust):
                if time.time() - start > time_limit:
                    return best_result
                    
                cur_f = current[i]
                d_i = demands[i]
                best_delta = 0
                best_new_f = -1
                
                # Vectorized delta calculation for speed
                # delta = assign_cost[i, :] - assign_cost[i, cur_f]
                # + open_cost changes
                # open_cost change:
                # if fac_count[cur_f] == 1: -open_cost[cur_f]
                # if fac_count[new_f] == 0: +open_cost[new_f]
                
                base_delta = assign_cost[i, :] - assign_cost[i, cur_f]
                
                # Mask infeasible facilities
                feasible_mask = (load + d_i) <= capacities
                feasible_mask[cur_f] = False # Don't move to same
                
                if not np.any(feasible_mask):
                    continue
                    
                # Calculate open cost adjustment
                # This is tricky to vectorize cleanly because it depends on fac_count[new_f]==0
                # and fac_count[cur_f]==1.
                
                # Let's do a semi-vectorized approach
                # Start with base assignment cost difference
                deltas = base_delta.copy()
                
                # If moving from a facility that becomes empty, we save its open cost
                if fac_count[cur_f] == 1:
                    deltas -= open_cost[cur_f]
                    
                # If moving to a facility that is currently empty, we pay its open cost
                empty_facs = np.where(fac_count == 0)[0]
                if len(empty_facs) > 0:
                    deltas[empty_facs] += open_cost[empty_facs]
                    
                # Apply feasibility mask
                deltas[~feasible_mask] = np.inf
                
                min_idx = np.argmin(deltas)
                min_delta = deltas[min_idx]
                
                if min_delta < -1e-9:
                    best_new_f = min_idx
                    best_delta = min_delta
                    # Apply move
                    load[cur_f] -= d_i
                    load[best_new_f] += d_i
                    fac_count[cur_f] -= 1
                    fac_count[best_new_f] += 1
                    current[i] = best_new_f
                    current_cost += best_delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_result = current.copy()
                        
            # 2. Swap
            # To speed up, we can just iterate all pairs. n=100, pairs=4950.
            # We can vectorize this too?
            # For each pair (i, j), delta = (cost[i, f_j] + cost[j, f_i]) - (cost[i, f_i] + cost[j, f_j])
            # Capacity checks:
            # load[f_i] - d_i + d_j <= cap[f_i]
            # load[f_j] - d_j + d_i <= cap[f_j]
            
            # Let's do a simple loop but break early if time is tight.
            # We can optimize by only checking pairs where f_i != f_j.
            
            for i in range(n_cust):
                if time.time() - start > time_limit:
                    return best_result
                f_i = current[i]
                d_i = demands[i]
                
                for j in range(i + 1, n_cust):
                    f_j = current[j]
                    if f_i == f_j:
                        continue
                    d_j = demands[j]
                    
                    # Check capacity after swap
                    if load[f_i] - d_i + d_j > capacities[f_i]:
                        continue
                    if load[f_j] - d_j + d_i > capacities[f_j]:
                        continue
                    
                    delta = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
                    
                    # Open cost change for swap between different facilities is 0
                    # (counts stay same)
                    
                    if delta < -1e-9:
                        # Apply swap
                        load[f_i] = load[f_i] - d_i + d_j
                        load[f_j] = load[f_j] - d_j + d_i
                        current[i] = f_j
                        current[j] = f_i
                        current_cost += delta
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_result = current.copy()
                        # Continue searching for more swaps in this pass
                        # Don't break, just update state and continue to next j
                        # Note: updating load/current inside the loop is safe as long as we
                        # re-fetch f_i, f_j, d_i, d_j for subsequent iterations?
                        # Actually, if we swap i and j, their assignments change.
                        # The inner loop continues with j+1.
                        # The current[i] is now f_j.
                        # The current[j] is now f_i.
                        # This is fine.
                        
        return best_result

    # Add Lagrangian Relaxation to generate more diverse candidates
    def lagrangian_relaxation(time_limit):
        start = time.time()
        multipliers = np.zeros(n_fac)
        best_solution = None
        best_cost = float('inf')
        
        # Subgradient method
        step_size = 1.0
        for k in range(100):
            if time.time() - start > time_limit:
                break
                
            # Solve subproblem: min sum_i,j (assign_cost[i,j] + multipliers[j]) x[i,j] + sum_j open_cost[j] y[j]
            # s.t. sum_j x[i,j] = 1, x[i,j] <= y[j]
            # This decouples if we ignore capacity.
            # For each customer i, choose j minimizing assign_cost[i,j] + multipliers[j].
            # But we must respect x[i,j] <= y[j].
            # If we set y[j]=1 for all j, it's just argmin.
            # Then we check capacity. If violated, update multipliers.
            
            # Simple assignment ignoring capacity but respecting open/close?
            # Actually, if we ignore capacity, we can just assign each customer to min cost facility.
            # But we need to pay open cost.
            # Let's solve the relaxed problem where y[j] is free (0 or 1).
            # Actually, standard Lagrangian relaxation of capacity constraints:
            # L(mu) = min sum_i,j (c_ij + mu_j) x_ij + sum_j f_j y_j
            # s.t. sum_j x_ij = 1, x_ij <= y_j
            
            # For a fixed mu, the problem decomposes by facility? No, by customer?
            # No, because y_j is shared.
            # However, we can solve it as a min-cost flow or just greedy?
            # Since x_ij <= y_j, if we assign to j, we pay f_j.
            # This is a facility location problem without capacity.
            # We can solve it by considering all subsets? No.
            # We can solve it by iterating?
            
            # Let's just use the greedy assignment with modified costs.
            # c'_ij = c_ij + mu_j.
            # Greedy: assign each customer to min c'_ij.
            # Then determine which facilities are opened (those with at least one customer).
            # This is a valid feasible solution for the relaxed problem?
            # Yes, if we open all facilities used.
            
            modified_cost = assign_cost + multipliers[np.newaxis, :]
            assignment = np.argmin(modified_cost, axis=1)
            
            # Check capacity
            load = np.zeros(n_fac)
            for i, f in enumerate(assignment):
                load[f] += demands[i]
                
            if np.all(load <= capacities + 1e-9):
                cost = compute_cost(assignment)
                if cost < best_cost:
                    best_cost = cost
                    best_solution = assignment.copy()
            else:
                # Update multipliers
                # g = load - capacities
                g = load - capacities
                g[g < 0] = 0 # Only positive violations contribute to subgradient?
                # Actually subgradient is load - capacities.
                # If load > cap, increase mu. If load < cap, decrease mu?
                # Standard: mu_j <- max(0, mu_j + step * (load_j - cap_j))
                
                # Step size adjustment
                # Use Armijo or fixed?
                # Let's try fixed step with decay.
                step = 10.0 / (k + 1)
                
                multipliers += step * g
                multipliers[multipliers < 0] = 0
                
        return best_solution

    # Run Lagrangian Relaxation
    if time.time() < deadline - 20:
        lr_solution = lagrangian_relaxation(time_limit=5.0)
        if lr_solution is not None:
            candidates.append(lr_solution)
            
    # Re-pick best initial candidate after adding LR
    best_init = min(candidates, key=compute_cost)

    remaining_time = deadline - time.time()
    if remaining_time > 10:
        # Run local search with a generous time limit
        final_result = _local_search(best_init, remaining_time - 5)
    else:
        final_result = best_init
        
    # Final Validation
    load_final = np.zeros(n_fac)
    for i, f in enumerate(final_result):
        if f < 0 or f >= n_fac:
            return best_init # Fallback to valid initial
        load_final[f] += demands[i]
        
    if np.any(load_final > capacities + 1e-9):
        return best_init
        
    return final_result
