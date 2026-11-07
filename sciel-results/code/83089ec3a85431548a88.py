"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility using a greedy + local search approach.
    
    Strategy:
    1. Greedy assignment: sort customers by demand descending, assign to cheapest
       feasible facility considering capacity.
    2. Local search: try to improve by moving customers to other facilities or
       swapping pairs of customers between facilities.
    """
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    # Greedy initialization: largest demand first, cheapest feasible facility
    assignment = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities, dtype=np.float64)
    used = np.zeros(n_facilities, dtype=bool)
    
    order = np.argsort(-demands)
    for i in order:
        # Sort facilities by cost for this customer
        fac_order = np.argsort(assign_cost[i])
        for f in fac_order:
            if load[f] + demands[i] <= capacities[f] + 1e-9:
                assignment[i] = f
                load[f] += demands[i]
                used[f] = True
                break
        else:
            # If no feasible facility found (shouldn't happen), assign to least loaded
            f = np.argmin(load)
            assignment[i] = f
            load[f] += demands[i]
            used[f] = True
    
    def total_cost(assignment, load, used):
        """Compute total cost: open costs + assignment costs."""
        open_cost_sum = np.sum(open_cost[used])
        assign_cost_sum = np.sum(assign_cost[np.arange(n_customers), assignment])
        return open_cost_sum + assign_cost_sum
    
    current_cost = total_cost(assignment, load, used)
    best_assignment = assignment.copy()
    best_load = load.copy()
    best_used = used.copy()
    best_cost = current_cost
    
    # Local search: try moving each customer to a different facility
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        # Try moving each customer
        for i in range(n_customers):
            if time.time() > deadline:
                break
            cur_f = assignment[i]
            # Try all other facilities
            for new_f in range(n_facilities):
                if new_f == cur_f:
                    continue
                # Check capacity
                if load[new_f] + demands[i] <= capacities[new_f] + 1e-9:
                    # Compute delta in cost
                    # Remove i from cur_f, add to new_f
                    delta_assign = assign_cost[i, new_f] - assign_cost[i, cur_f]
                    delta_open = 0.0
                    if not used[new_f]:
                        delta_open += open_cost[new_f]
                    if load[cur_f] - demands[i] <= 1e-9:
                        # cur_f becomes empty
                        delta_open -= open_cost[cur_f]
                    
                    new_cost = current_cost + delta_assign + delta_open
                    if new_cost < current_cost - 1e-9:
                        # Apply move
                        load[cur_f] -= demands[i]
                        if load[cur_f] <= 1e-9:
                            load[cur_f] = 0.0
                            used[cur_f] = False
                        load[new_f] += demands[i]
                        used[new_f] = True
                        assignment[i] = new_f
                        current_cost = new_cost
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_assignment = assignment.copy()
                            best_load = load.copy()
                            best_used = used.copy()
    
    # Second phase: try swapping pairs of customers
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(n_customers):
            if time.time() > deadline:
                break
            for j in range(i + 1, n_customers):
                if time.time() > deadline:
                    break
                fi = assignment[i]
                fj = assignment[j]
                if fi == fj:
                    continue
                # Check if swapping is feasible
                if (load[fi] - demands[i] + demands[j] <= capacities[fi] + 1e-9 and
                    load[fj] - demands[j] + demands[i] <= capacities[fj] + 1e-9):
                    delta_assign = (assign_cost[i, fj] + assign_cost[j, fi]) - \
                                   (assign_cost[i, fi] + assign_cost[j, fj])
                    delta_open = 0.0
                    # After swap, fi has j, fj has i
                    # Check if any facility becomes empty
                    if load[fi] - demands[i] + demands[j] <= 1e-9:
                        # fi becomes empty (only if j demand is 0, unlikely)
                        pass
                    if load[fj] - demands[j] + demands[i] <= 1e-9:
                        pass
                    # No open cost change unless a facility becomes empty
                    # More carefully:
                    new_load_fi = load[fi] - demands[i] + demands[j]
                    new_load_fj = load[fj] - demands[j] + demands[i]
                    if load[fi] > 1e-9 and new_load_fi <= 1e-9:
                        delta_open -= open_cost[fi]
                    if load[fj] > 1e-9 and new_load_fj <= 1e-9:
                        delta_open -= open_cost[fj]
                    
                    new_cost = current_cost + delta_assign + delta_open
                    if new_cost < current_cost - 1e-9:
                        # Apply swap
                        load[fi] -= demands[i]
                        load[fj] -= demands[j]
                        load[fi] += demands[j]
                        load[fj] += demands[i]
                        assignment[i] = fj
                        assignment[j] = fi
                        current_cost = new_cost
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_assignment = assignment.copy()
                            best_load = load.copy()
                            best_used = used.copy()
    
    # Return best found
    result = best_assignment.astype(int)
    return result
# EVOLVE-BLOCK-END
