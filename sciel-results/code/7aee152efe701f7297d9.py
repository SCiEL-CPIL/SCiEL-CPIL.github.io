import numpy as np
import time
import random
from scipy.optimize import linprog

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    demands = np.array(demands, dtype=float)
    capacities = np.array(capacities, dtype=float)
    open_cost = np.array(open_cost, dtype=float)
    assign_cost = np.array(assign_cost, dtype=float)
    
    # --- 1. LP Relaxation for Initialization ---
    c_obj = np.concatenate([assign_cost.flatten(), open_cost])
    bounds = [(0, 1)] * (n_cust * n_fac + n_fac)
    
    A_eq = np.zeros((n_cust, n_cust * n_fac + n_fac))
    for i in range(n_cust):
        for j in range(n_fac):
            A_eq[i, i * n_fac + j] = 1.0
    b_eq = np.ones(n_cust)
    
    n_cons_ub = n_fac + n_cust * n_fac
    A_ub = np.zeros((n_cons_ub, n_cust * n_fac + n_fac))
    b_ub = np.zeros(n_cons_ub)
    
    for j in range(n_fac):
        row_idx = j
        for i in range(n_cust):
            A_ub[row_idx, i * n_fac + j] = demands[i]
        A_ub[row_idx, n_cust * n_fac + j] = -capacities[j]
        
    for i in range(n_cust):
        for j in range(n_fac):
            row_idx = n_fac + i * n_fac + j
            A_ub[row_idx, i * n_fac + j] = 1.0
            A_ub[row_idx, n_cust * n_fac + j] = -1.0
            
    lp_solution = None
    try:
        if time.time() < deadline - 10:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if res.status == 0:
                lp_solution = res.x
    except Exception:
        lp_solution = None
        
    # --- 2. Initialization ---
    def compute_cost(result):
        load = np.zeros(n_fac)
        fac_count = np.zeros(n_fac, dtype=int)
        for i, f in enumerate(result):
            load[f] += demands[i]
            fac_count[f] += 1
        if np.any(load > capacities + 1e-9):
            return float('inf')
        return np.sum(open_cost[fac_count > 0]) + np.sum(assign_cost[np.arange(n_cust), result])

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
                remaining = capacities - load
                f = np.argmax(remaining)
                if remaining[f] >= demands[i]:
                    load[f] += demands[i]
                    result[i] = f
                else:
                    return None
            
        return result

    def lp_rounding_init(lp_sol):
        if lp_sol is None:
            return None
        x_lp = lp_sol[:n_cust * n_fac].reshape(n_cust, n_fac)
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        
        order = np.argsort(-demands)
        
        for i in order:
            sorted_facs = np.argsort(-x_lp[i])
            assigned = False
            for f in sorted_facs:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break
            
            if not assigned:
                feasible = [f for f in range(n_fac) if load[f] + demands[i] <= capacities[f]]
                if feasible:
                    f_best = min(feasible, key=lambda f: assign_cost[i, f])
                    load[f_best] += demands[i]
                    result[i] = f_best
                else:
                    return None
        
        return result

    candidates = []
    
    # 1. LP Rounding
    init_lp = lp_rounding_init(lp_solution)
    if init_lp is not None:
        candidates.append(init_lp)
        
    # 2. Greedy Largest First
    init_g1 = greedy_init(lambda d: -d)
    if init_g1 is not None:
        candidates.append(init_g1)
        
    # 3. Greedy Smallest First
    init_g2 = greedy_init(lambda d: d)
    if init_g2 is not None:
        candidates.append(init_g2)
        
    # 4. Randomized Greedy (a few times)
    for _ in range(3):
        if time.time() > deadline - 20:
            break
        init_r = greedy_init(lambda d: -d, randomize=True)
        if init_r is not None:
            candidates.append(init_r)
            
    if not candidates:
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        for i in range(n_cust):
            for f in range(n_fac):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break
        candidates.append(result)

    best_init = min(candidates, key=compute_cost)
    
    # --- 3. Enhanced Local Search with Open/Close Moves ---
    
    def _local_search(result, demands, capacities, open_cost, assign_cost, time_limit):
        n_cust = len(demands)
        n_fac = len(capacities)
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
        
        while improved and (time.time() - start < time_limit):
            improved = False
            
            # 1. Single Reassignment
            for i in range(n_cust):
                if time.time() - start > time_limit:
                    return best_result
                cur_f = current[i]
                d_i = demands[i]
                best_delta = 0
                best_new_f = -1
                
                for new_f in range(n_fac):
                    if new_f == cur_f:
                        continue
                    if load[new_f] + d_i > capacities[new_f]:
                        continue
                    
                    delta = assign_cost[i, new_f] - assign_cost[i, cur_f]
                    
                    if fac_count[cur_f] == 1:
                        delta -= open_cost[cur_f]
                    if fac_count[new_f] == 0:
                        delta += open_cost[new_f]
                        
                    if delta < best_delta:
                        best_delta = delta
                        best_new_f = new_f
                        
                if best_new_f != -1:
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
                    
                    if load[f_i] - d_i + d_j > capacities[f_i]:
                        continue
                    if load[f_j] - d_j + d_i > capacities[f_j]:
                        continue
                    
                    delta = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
                    
                    if delta < -1e-9:
                        load[f_i] = load[f_i] - d_i + d_j
                        load[f_j] = load[f_j] - d_j + d_i
                        current[i] = f_j
                        current[j] = f_i
                        current_cost += delta
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_result = current.copy()
                        break
                        
            # 3. Close-Facility Move (KEY ENHANCEMENT from inspiration)
            # Try closing each open facility and reassigning its customers optimally
            for close_f in range(n_fac):
                if time.time() - start > time_limit:
                    return best_result
                if fac_count[close_f] == 0:
                    continue
                    
                # Collect customers at this facility
                customers_to_reassign = [i for i in range(n_cust) if current[i] == close_f]
                if not customers_to_reassign:
                    continue
                
                # Try to reassign all these customers to other open facilities
                # using a greedy approach (sort by demand descending, assign to cheapest feasible)
                trial_load = load.copy()
                trial_fac_count = fac_count.copy()
                trial_current = current.copy()
                
                # Remove close_f from consideration
                trial_load[close_f] = 0
                trial_fac_count[close_f] = 0
                
                # Sort customers by demand descending
                customers_to_reassign.sort(key=lambda i: -demands[i])
                
                success = True
                reassign_cost_delta = 0.0
                for i in customers_to_reassign:
                    d_i = demands[i]
                    # Find cheapest feasible facility (excluding close_f)
                    best_new_f = -1
                    best_cost_i = float('inf')
                    for new_f in range(n_fac):
                        if new_f == close_f:
                            continue
                        if trial_load[new_f] + d_i <= capacities[new_f]:
                            c = assign_cost[i, new_f]
                            if c < best_cost_i:
                                best_cost_i = c
                                best_new_f = new_f
                    
                    if best_new_f == -1:
                        success = False
                        break
                    
                    trial_load[best_new_f] += d_i
                    trial_fac_count[best_new_f] += 1
                    trial_current[i] = best_new_f
                    reassign_cost_delta += assign_cost[i, best_new_f] - assign_cost[i, close_f]
                
                if not success:
                    continue
                
                # Calculate total delta
                # Save open_cost for close_f
                delta = reassign_cost_delta - open_cost[close_f]
                
                # Check if any new facilities were opened (shouldn't happen since we're closing one)
                # But we might open facilities that were previously empty
                for new_f in range(n_fac):
                    if new_f != close_f and fac_count[new_f] == 0 and trial_fac_count[new_f] > 0:
                        delta += open_cost[new_f]
                
                if delta < -1e-9:
                    # Apply the close move
                    load = trial_load
                    fac_count = trial_fac_count
                    current = trial_current
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_result = current.copy()
                    
                    # After closing, do a quick single-pass reassignment to further improve
                    for i in customers_to_reassign:
                        cur_f = current[i]
                        d_i = demands[i]
                        best_delta = 0
                        best_new_f = -1
                        for new_f in range(n_fac):
                            if new_f == cur_f:
                                continue
                            if load[new_f] + d_i > capacities[new_f]:
                                continue
                            delta2 = assign_cost[i, new_f] - assign_cost[i, cur_f]
                            if fac_count[cur_f] == 1:
                                delta2 -= open_cost[cur_f]
                            if fac_count[new_f] == 0:
                                delta2 += open_cost[new_f]
                            if delta2 < best_delta:
                                best_delta = delta2
                                best_new_f = new_f
                        if best_new_f != -1:
                            load[cur_f] -= d_i
                            load[best_new_f] += d_i
                            fac_count[cur_f] -= 1
                            fac_count[best_new_f] += 1
                            current[i] = best_new_f
                            current_cost += best_delta
                            if current_cost < best_cost:
                                best_cost = current_cost
                                best_result = current.copy()
        
        return best_result

    remaining_time = deadline - time.time()
    if remaining_time > 5:
        final_result = _local_search(best_init, demands, capacities, open_cost, assign_cost, remaining_time - 2)
    else:
        final_result = best_init
        
    # Final Validation
    load_final = np.zeros(n_fac)
    for i, f in enumerate(final_result):
        if f < 0 or f >= n_fac:
            return np.zeros(n_cust, dtype=int)
        load_final[f] += demands[i]
        
    if np.any(load_final > capacities + 1e-9):
        return best_init
        
    return final_result
