import numpy as np
import time
import random
from scipy.optimize import milp, LinearConstraint, Bounds

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s limit

    n_customers = len(demands)
    n_facilities = len(capacities)
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    best_result = None
    best_cost = float('inf')

    def calc_cost(res):
        """Calculate total cost of an assignment."""
        if res is None:
            return float('inf')
        res = np.asarray(res, dtype=int)
        if len(res) != n_customers:
            return float('inf')
        if np.any(res < 0) or np.any(res >= n_facilities):
            return float('inf')
        
        load = np.zeros(n_facilities)
        np.add.at(load, res, demands)
        if np.any(load > capacities + 1e-9):
            return float('inf')
            
        used = load > 0
        total = np.sum(open_cost[used])
        total += np.sum(assign_cost[np.arange(n_customers), res])
        return total

    def greedy_init(seed=0):
        """Greedy initialization with optional randomization."""
        rng = np.random.default_rng(seed)
        res = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        
        # Process larger demands first to improve feasibility
        order = np.argsort(-demands)
        
        for i in order:
            fac_costs = assign_cost[i]
            if seed > 0:
                # Add small noise to break ties and diversify
                noise = rng.uniform(0, 0.1 * np.std(fac_costs), n_facilities)
                noisy_costs = fac_costs + noise
                fac_order = np.argsort(noisy_costs)
            else:
                fac_order = np.argsort(fac_costs)
            
            assigned = False
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    res[i] = f
                    load[f] += demands[i]
                    assigned = True
                    break
            
            if not assigned:
                # Fallback: assign to facility with most remaining space
                space = capacities - load
                max_space = np.max(space)
                candidates = np.where(space >= max_space - 1e-9)[0]
                if len(candidates) > 0:
                    f = rng.choice(candidates)
                    res[i] = f
                    load[f] += demands[i]
                else:
                    # Should not happen if total capacity is sufficient, but handle gracefully
                    f = 0
                    res[i] = f
                    load[f] += demands[i]
                    
        return res

    def local_search(res, load, max_iters=500):
        """Local search with relocate and swap moves."""
        res = res.copy()
        load = load.copy()
        current_cost = calc_cost(res)
        
        if current_cost == float('inf'):
            return res, current_cost

        improved = True
        it = 0
        while improved and it < max_iters and time.time() < deadline:
            improved = False
            it += 1
            
            # --- Phase 1: Relocate Moves (First Improvement or Best Improvement) ---
            # We use a best-improvement strategy per pass for stability, 
            # but limit the search scope to keep it fast.
            best_delta_global = 1e-9
            best_move_global = None
            
            # Shuffle customers to avoid bias
            cust_order = np.random.permutation(n_customers)
            
            for i in cust_order:
                f_old = res[i]
                curr_cost_i = assign_cost[i, f_old]
                
                for f_new in range(n_facilities):
                    if f_new == f_old:
                        continue
                    if load[f_new] + demands[i] > capacities[f_new] + 1e-9:
                        continue
                    
                    delta = assign_cost[i, f_new] - curr_cost_i
                    
                    # Opening cost adjustments
                    if load[f_old] == demands[i]:
                        # This customer is the only one at f_old, so closing it saves open_cost
                        delta -= open_cost[f_old]
                    if load[f_new] == 0:
                        # Opening f_new adds open_cost
                        delta += open_cost[f_new]
                    
                    if delta < best_delta_global:
                        best_delta_global = delta
                        best_move_global = (i, f_new)
            
            if best_move_global is not None:
                i, f_new = best_move_global
                f_old = res[i]
                load[f_old] -= demands[i]
                load[f_new] += demands[i]
                res[i] = f_new
                current_cost += best_delta_global
                improved = True
            
            # --- Phase 2: Swap Moves ---
            # Try a limited number of random swaps to escape local minima
            if it % 5 == 0:
                swapped = False
                # Sample a subset of pairs to keep it fast
                num_swaps_to_try = 50
                for _ in range(num_swaps_to_try):
                    if time.time() > deadline:
                        break
                    i = np.random.randint(n_customers)
                    j = np.random.randint(n_customers)
                    if i == j:
                        continue
                    
                    f_i = res[i]
                    f_j = res[j]
                    if f_i == f_j:
                        continue
                    
                    # Check capacity constraints for swap
                    if load[f_i] - demands[i] + demands[j] > capacities[f_i] + 1e-9:
                        continue
                    if load[f_j] - demands[j] + demands[i] > capacities[f_j] + 1e-9:
                        continue
                    
                    delta = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
                    
                    # Note: Swap moves do not change the set of open facilities, so no open_cost change
                    if delta < -1e-9:
                        load[f_i] += demands[j] - demands[i]
                        load[f_j] += demands[i] - demands[j]
                        res[i], res[j] = f_j, f_i
                        current_cost += delta
                        improved = True
                        swapped = True
                        break
                # If a swap happened, we don't necessarily need to break the whole while loop,
                # but it helps to re-evaluate relocate moves in the next iteration.
        
        return res, current_cost

    def solve_milp():
        """
        Solve the capacitated facility location problem as a MILP.
        Returns the best solution found or None if it fails/times out.
        """
        if time.time() > deadline - 5:
            return None

        n_x = n_customers * n_facilities
        n_y = n_facilities
        n_vars = n_x + n_y
        
        # Objective coefficients
        c = np.zeros(n_vars)
        for i in range(n_customers):
            for f in range(n_facilities):
                c[i * n_facilities + f] = assign_cost[i, f]
        for f in range(n_facilities):
            c[n_x + f] = open_cost[f]
        
        integrality = np.ones(n_vars)
        bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
        
        # Constraints
        A_rows = []
        lb_list = []
        ub_list = []
        
        # 1. Assignment constraint: sum_f x[i,f] = 1
        for i in range(n_customers):
            row = np.zeros(n_vars)
            for f in range(n_facilities):
                row[i * n_facilities + f] = 1.0
            A_rows.append(row)
            lb_list.append(1.0)
            ub_list.append(1.0)
        
        # 2. Capacity constraint: sum_i demands[i] * x[i,f] <= capacities[f] * y[f]
        # Rearranged: sum_i demands[i] * x[i,f] - capacities[f] * y[f] <= 0
        for f in range(n_facilities):
            row = np.zeros(n_vars)
            for i in range(n_customers):
                row[i * n_facilities + f] = demands[i]
            row[n_x + f] = -capacities[f]
            A_rows.append(row)
            lb_list.append(-np.inf)
            ub_list.append(0.0)
        
        # 3. Linking constraint: x[i,f] <= y[f]
        # Rearranged: x[i,f] - y[f] <= 0
        for i in range(n_customers):
            for f in range(n_facilities):
                row = np.zeros(n_vars)
                row[i * n_facilities + f] = 1.0
                row[n_x + f] = -1.0
                A_rows.append(row)
                lb_list.append(-np.inf)
                ub_list.append(0.0)
        
        A = np.array(A_rows)
        lb = np.array(lb_list)
        ub = np.array(ub_list)
        
        try:
            res = milp(
                c=c,
                constraints=LinearConstraint(A, lb, ub),
                integrality=integrality,
                bounds=bounds,
                options={"time_limit": max(1, int(deadline - time.time() - 5))}
            )
            
            if res.status == 0:
                x = res.x[:n_x].reshape(n_customers, n_facilities)
                assignment = np.argmax(x, axis=1)
                return assignment
            else:
                return None
        except Exception:
            return None

    # --- Strategy Execution ---
    
    # 1. Try MILP first as it has the potential for global optimality.
    #    We allocate a significant chunk of time to it.
    milp_start = time.time()
    milp_result = solve_milp()
    milp_elapsed = time.time() - milp_start
    
    if milp_result is not None:
        cost = calc_cost(milp_result)
        if cost < best_cost:
            best_cost = cost
            best_result = milp_result.copy()
            
            # If MILP solved quickly, we can try to improve with local search?
            # Usually MILP is good, but local search might polish it if it's not strictly optimal due to time limit.
            if milp_elapsed < 20 and time.time() < deadline - 10:
                load = np.zeros(n_facilities)
                np.add.at(load, milp_result, demands)
                polished, polished_cost = local_search(milp_result, load, max_iters=100)
                if polished_cost < best_cost:
                    best_cost = polished_cost
                    best_result = polished.copy()
    else:
        # MILP failed or timed out. Fall back to heuristics.
        pass

    # 2. Heuristic Fallback / Improvement
    # If we have time, run greedy + local search with a few restarts.
    # This is especially important if MILP didn't run or didn't find a good solution.
    
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        seed = r
        res_r = greedy_init(seed)
        cost_r = calc_cost(res_r)
        
        if cost_r == float('inf'):
            continue
            
        # Refine with local search
        load_r = np.zeros(n_facilities)
        np.add.at(load_r, res_r, demands)
        
        refined_res, refined_cost = local_search(res_r, load_r, max_iters=200)
        
        if refined_cost < best_cost:
            best_cost = refined_cost
            best_result = refined_res.copy()

    # Final safety check
    if best_result is None:
        # Absolute fallback: assign everyone to facility 0 (might be infeasible, but better than crash)
        # Or try a simple greedy without randomization
        res_fallback = greedy_init(0)
        cost_fallback = calc_cost(res_fallback)
        if cost_fallback < float('inf'):
            return res_fallback
        # If even that fails, return zeros (will likely be invalid, but prevents exception)
        return np.zeros(n_customers, dtype=int)

    return best_result
