import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    def get_cost(res, ld):
        total = 0.0
        for f in range(n_facilities):
            if ld[f] > 0:
                total += open_cost[f]
        total += np.sum(assign_cost[np.arange(n_customers), res])
        return total

    def try_move(i, f_new, res, ld):
        f_old = res[i]
        if f_new == f_old:
            return 0.0, False
        if ld[f_new] + demands[i] > capacities[f_new]:
            return 0.0, False
        
        delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
        delta_open = 0.0
        
        if ld[f_old] == demands[i]:
            delta_open -= open_cost[f_old]
        if ld[f_new] == 0:
            delta_open += open_cost[f_new]
            
        return delta_assign + delta_open, True

    def do_move(i, f_new, res, ld):
        f_old = res[i]
        ld[f_old] -= demands[i]
        ld[f_new] += demands[i]
        res[i] = f_new

    def try_swap(i, j, res, ld):
        f_i = res[i]
        f_j = res[j]
        if f_i == f_j:
            return 0.0, False
        
        if ld[f_i] - demands[i] + demands[j] > capacities[f_i]:
            return 0.0, False
        if ld[f_j] - demands[j] + demands[i] > capacities[f_j]:
            return 0.0, False
            
        delta_assign = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
        return delta_assign, True

    def do_swap(i, j, res, ld):
        f_i = res[i]
        f_j = res[j]
        ld[f_i] -= demands[i]
        ld[f_j] -= demands[j]
        ld[f_i] += demands[j]
        ld[f_j] += demands[i]
        res[i] = f_j
        res[j] = f_i

    def local_search(res, ld, current_cost):
        improved = True
        while improved:
            improved = False
            
            # 1. Best Improvement Move
            best_delta = 0.0
            best_i = -1
            best_f = -1
            for i in range(n_customers):
                f_old = res[i]
                for f_new in range(n_facilities):
                    if f_new == f_old:
                        continue
                    delta, valid = try_move(i, f_new, res, ld)
                    if valid and delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_f = f_new
            
            if best_i != -1:
                do_move(best_i, best_f, res, ld)
                current_cost += best_delta
                improved = True
                continue

            # 2. Best Improvement Swap
            best_delta_swap = 0.0
            best_i_swap = -1
            best_j_swap = -1
            for i in range(n_customers):
                f_i = res[i]
                for j in range(i + 1, n_customers):
                    f_j = res[j]
                    if f_i == f_j:
                        continue
                    delta, valid = try_swap(i, j, res, ld)
                    if valid and delta < best_delta_swap:
                        best_delta_swap = delta
                        best_i_swap = i
                        best_j_swap = j
                        
            if best_i_swap != -1:
                do_swap(best_i_swap, best_j_swap, res, ld)
                current_cost += best_delta_swap
                improved = True

        return res, ld, current_cost

    def construct_solution():
        res = np.zeros(n_customers, dtype=int)
        ld = np.zeros(n_facilities)
        unassigned = list(range(n_customers))
        
        # Sort customers by demand descending to prioritize large ones
        unassigned.sort(key=lambda x: -demands[x])
        
        for i in unassigned:
            # Find feasible facilities
            feasible = []
            for f in range(n_facilities):
                if ld[f] + demands[i] <= capacities[f]:
                    feasible.append(f)
            
            if not feasible:
                # Fallback: should not happen if instance is feasible
                # Assign to facility with most remaining capacity
                f_best = np.argmax(capacities - ld)
                res[i] = f_best
                ld[f_best] += demands[i]
                continue
            
            # Sort feasible by assignment cost
            feasible.sort(key=lambda f: assign_cost[i, f])
            
            # RCL: Take top 5 or 20% of feasible
            k = max(1, min(5, len(feasible)))
            rcl = feasible[:k]
            
            # Random choice from RCL
            f_chosen = rcl[np.random.randint(len(rcl))]
            
            res[i] = f_chosen
            ld[f_chosen] += demands[i]
            
        return res, ld

    start_time = time.time()
    time_limit = 18.0 # Slightly higher to allow more GRASP iterations
    
    best_result = None
    best_load = None
    best_cost = float('inf')
    
    # Run GRASP iterations
    while time.time() - start_time < time_limit:
        res, ld = construct_solution()
        cost = get_cost(res, ld)
        
        # Local Search
        res, ld, cost = local_search(res, ld, cost)
        
        if cost < best_cost:
            best_cost = cost
            best_result = res.copy()
            best_load = ld.copy()
            
    return best_result
