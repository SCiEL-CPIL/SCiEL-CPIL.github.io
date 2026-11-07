"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def assign(demands, capacities, open_cost, assign_cost):
    n_cust = len(demands)
    n_fac = len(capacities)
    deadline = time.time() + 110  # Safety margin before 120s limit

    # Sort customers by demand descending (largest first for greedy)
    order = np.argsort(-demands)
    demands_sorted = demands[order]
    
    # Greedy assignment: largest demand first, cheapest feasible facility
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac, dtype=int)
    
    for i in order:
        # Sort facilities by assignment cost for this customer
        fac_order = np.argsort(assign_cost[i])
        for f in fac_order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
        else:
            # Fallback: assign to facility with most remaining capacity
            remaining = capacities - load
            f = np.argmax(remaining)
            load[f] += demands[i]
            result[i] = f
    
    # Local search: try to improve by reassigning customers
    def compute_cost(res):
        total = 0.0
        used = np.zeros(n_fac, dtype=bool)
        for i in range(n_cust):
            f = res[i]
            total += assign_cost[i, f]
            used[f] = True
        total += np.sum(open_cost[used])
        return total
    
    best_result = result.copy()
    best_cost = compute_cost(result)
    
    # Local search with time limit
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try moving each customer to a different facility
        for i in range(n_cust):
            if time.time() > deadline:
                break
            current_f = best_result[i]
            # Sort other facilities by cost
            other_facs = np.argsort(assign_cost[i])
            for f in other_facs:
                if f == current_f:
                    continue
                if load[f] + demands[i] <= capacities[f]:
                    # Calculate delta
                    old_open = open_cost[current_f] if np.all(best_result != current_f) or True else 0
                    # Check if current_f becomes empty
                    count_current = np.sum(best_result == current_f)
                    count_f = np.sum(best_result == f)
                    
                    delta_assign = assign_cost[i, f] - assign_cost[i, current_f]
                    delta_open = 0.0
                    if count_current == 1:
                        delta_open -= open_cost[current_f]
                    if count_f == 0:
                        delta_open += open_cost[f]
                    
                    total_delta = delta_assign + delta_open
                    if total_delta < -1e-9:
                        # Accept improvement
                        load[current_f] -= demands[i]
                        load[f] += demands[i]
                        best_result[i] = f
                        best_cost += total_delta
                        improved = True
                        break
    
    # Verify feasibility
    final_load = np.zeros(n_fac, dtype=int)
    for i in range(n_cust):
        final_load[best_result[i]] += demands[i]
    if np.any(final_load > capacities):
        # Fallback to initial greedy result
        best_result = result
    
    return best_result.astype(int)
# EVOLVE-BLOCK-END
