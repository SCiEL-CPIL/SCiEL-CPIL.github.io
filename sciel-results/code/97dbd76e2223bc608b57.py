# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility respecting capacities.
    Minimizes open_cost (for open facilities) + assign_cost.
    Strategy: Improved greedy with open-cost awareness, fast local search,
    and cluster reoptimization.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    start_time = time.time()
    deadline = start_time + 140.0

    # Precompute sorted facility indices by assignment cost
    fac_order = np.argsort(assign_cost, axis=1)

    def _compute_obj(result):
        """Compute total objective for a valid assignment."""
        if result is None:
            return float('inf')
        load = np.zeros(n_fac, dtype=np.int64)
        valid = True
        for i in range(n_cust):
            f = result[i]
            if f < 0 or f >= n_fac:
                valid = False
                break
            load[f] += demands[i]
        if not valid or np.any(load > capacities):
            return float('inf')
        open_facs = load > 0
        total = np.sum(open_cost[open_facs]) + np.sum(assign_cost[np.arange(n_cust), result])
        return total

    def _greedy_assign(order, f_order):
        """
        Greedy assignment: process customers in order,
        assign each to the best facility considering both assign cost and open cost impact.
        """
        load = np.zeros(n_fac, dtype=np.int64)
        result = np.full(n_cust, -1, dtype=np.int64)
        for i in order:
            best_f = -1
            best_delta = float('inf')
            for f in f_order[i]:
                if load[f] + demands[i] > capacities[f]:
                    continue
                assign_c = assign_cost[i, f]
                open_delta = 0.0
                if load[f] == 0:
                    open_delta = open_cost[f]
                delta = assign_c + open_delta
                if delta < best_delta:
                    best_delta = delta
                    best_f = f
            if best_f >= 0:
                load[best_f] += demands[i]
                result[i] = best_f
            else:
                for f in range(n_fac):
                    if load[f] + demands[i] <= capacities[f]:
                        load[f] += demands[i]
                        result[i] = f
                        break
                else:
                    result[i] = 0
                    load[0] += demands[i]
        return result, load

    def _local_search_fast(result, load, time_limit):
        """
        Fast local search: try 1-moves with efficient delta computation.
        """
        current_obj = _compute_obj(result)
        best_obj = current_obj
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            order = np.random.permutation(n_cust)
            for i in order:
                if time.time() >= time_limit:
                    break
                current_f = result[i]
                if current_f < 0:
                    continue
                # Find best alternative facility
                best_f = -1
                best_delta = -1e-9
                for f in fac_order[i]:
                    if f == current_f:
                        continue
                    if load[f] + demands[i] > capacities[f]:
                        continue
                    delta_assign = assign_cost[i, f] - assign_cost[i, current_f]
                    delta_open = 0.0
                    if load[current_f] == demands[i]:
                        delta_open -= open_cost[current_f]
                    if load[f] == 0:
                        delta_open += open_cost[f]
                    delta = delta_assign + delta_open
                    if delta < best_delta:
                        best_delta = delta
                        best_f = f
                if best_f >= 0 and best_delta < -1e-9:
                    load[current_f] -= demands[i]
                    load[best_f] += demands[i]
                    result[i] = best_f
                    current_obj += best_delta
                    improved = True
                    if current_obj < best_obj:
                        best_obj = current_obj
        return result, load, best_obj

    def _cluster_reopt(result, load, time_limit):
        """
        For each open facility, try reassigning all its customers to other facilities.
        """
        current_obj = _compute_obj(result)
        best_obj = current_obj
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            open_facs = np.where(load > 0)[0]
            for f_open in open_facs:
                if time.time() >= time_limit:
                    break
                customers_here = np.where(result == f_open)[0]
                if len(customers_here) == 0:
                    continue
                new_load = load.copy()
                new_load[f_open] = 0
                new_result = result.copy()
                new_obj_delta = 0.0
                feasible = True
                for i in customers_here:
                    best_f = -1
                    best_d = float('inf')
                    for f in fac_order[i]:
                        if f == f_open:
                            continue
                        if new_load[f] + demands[i] <= capacities[f]:
                            d = assign_cost[i, f] - assign_cost[i, f_open]
                            if new_load[f] == 0 and load[f] == 0:
                                d += open_cost[f]
                            if d < best_d:
                                best_d = d
                                best_f = f
                    if best_f < 0:
                        feasible = False
                        break
                    new_load[best_f] += demands[i]
                    new_result[i] = best_f
                    new_obj_delta += best_d
                if not feasible:
                    continue
                # Account for open/close changes
                for f in range(n_fac):
                    if load[f] > 0 and new_load[f] == 0:
                        new_obj_delta -= open_cost[f]
                    elif load[f] == 0 and new_load[f] > 0:
                        new_obj_delta += open_cost[f]
                
                if new_obj_delta < -1e-9:
                    load = new_load
                    result = new_result
                    current_obj += new_obj_delta
                    improved = True
                    if current_obj < best_obj:
                        best_obj = current_obj
                    break
        return result, load, best_obj

    def _swap_search(result, load, time_limit):
        """
        Local search: try swapping assignments of two customers to reduce cost.
        """
        best_obj = _compute_obj(result)
        current_obj = best_obj
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try pairs of customers
            for i in range(n_cust):
                if time.time() >= time_limit:
                    break
                for j in range(i + 1, n_cust):
                    if time.time() >= time_limit:
                        break
                    fi = result[i]
                    fj = result[j]
                    if fi == fj:
                        continue
                    # Try swapping: i -> fj, j -> fi
                    # Check capacity
                    # After swap:
                    #   load[fi] changes by -demands[i] + demands[j]
                    #   load[fj] changes by -demands[j] + demands[i]
                    new_load_fi = load[fi] - demands[i] + demands[j]
                    new_load_fj = load[fj] - demands[j] + demands[i]
                    if new_load_fi < 0 or new_load_fj < 0:
                        continue
                    if new_load_fi > capacities[fi] or new_load_fj > capacities[fj]:
                        continue
                    # Compute delta
                    delta_assign = (assign_cost[i, fj] - assign_cost[i, fi]) + (assign_cost[j, fi] - assign_cost[j, fj])
                    delta_open = 0.0
                    # Check if any facility becomes empty or was empty
                    # fi: old load = load[fi], new load = new_load_fi
                    if load[fi] > 0 and new_load_fi == 0:
                        delta_open -= open_cost[fi]
                    elif load[fi] == 0 and new_load_fi > 0:
                        delta_open += open_cost[fi]
                    # fj: old load = load[fj], new load = new_load_fj
                    if load[fj] > 0 and new_load_fj == 0:
                        delta_open -= open_cost[fj]
                    elif load[fj] == 0 and new_load_fj > 0:
                        delta_open += open_cost[fj]
                    delta = delta_assign + delta_open
                    if delta < -1e-9:
                        # Apply swap
                        load[fi] = new_load_fi
                        load[fj] = new_load_fj
                        result[i] = fj
                        result[j] = fi
                        current_obj += delta
                        improved = True
                        break
                if improved:
                    break  # Restart outer loop
            if current_obj < best_obj - 1e-9:
                best_obj = current_obj
        return result, load, best_obj

    # Generate diverse initial solutions
    best_result = None
    best_load = None
    best_obj = float('inf')

    strategies = []
    strategies.append(np.argsort(-demands))
    strategies.append(np.argsort(demands))
    for _ in range(5):
        strategies.append(np.random.permutation(n_cust))
    for _ in range(3):
        strategies.append(np.argsort(assign_cost.min(axis=1)))

    for order in strategies:
        if time.time() >= deadline - 5:
            break
        result, load = _greedy_assign(order, fac_order)
        if np.any(load > capacities):
            continue
        obj = _compute_obj(result)
        if obj < best_obj:
            best_obj = obj
            best_result = result.copy()
            best_load = load.copy()

    if best_result is None:
        order = np.argsort(-demands)
        best_result, best_load = _greedy_assign(order, fac_order)
        best_obj = _compute_obj(best_result)

    # Phase 1: Fast local search (1-moves)
    if time.time() < deadline - 10:
        best_result, best_load, best_obj = _local_search_fast(best_result, best_load, deadline - 10)

    # Phase 2: Cluster reoptimization
    if time.time() < deadline - 5:
        best_result, best_load, best_obj = _cluster_reopt(best_result, best_load, deadline - 5)

    # Phase 3: Swap search
    if time.time() < deadline - 5:
        best_result, best_load, best_obj = _swap_search(best_result, best_load, deadline - 5)

    # Phase 4: Iterate
    while time.time() < deadline - 3:
        prev_obj = best_obj
        best_result, best_load, best_obj = _local_search_fast(best_result, best_load, deadline - 3)
        if time.time() < deadline - 3:
            best_result, best_load, best_obj = _cluster_reopt(best_result, best_load, deadline - 3)
        if time.time() < deadline - 3:
            best_result, best_load, best_obj = _swap_search(best_result, best_load, deadline - 3)
        if best_obj >= prev_obj - 1e-9:
            break

    # Final validation
    load = np.zeros(n_fac, dtype=np.int64)
    for i in range(n_cust):
        f = best_result[i]
        if f < 0 or f >= n_fac:
            f = 0
            best_result[i] = 0
        load[f] += demands[i]

    if np.any(load > capacities):
        order = np.argsort(-demands)
        best_result, best_load = _greedy_assign(order, fac_order)
        load = np.zeros(n_fac, dtype=np.int64)
        for i in range(n_cust):
            f = best_result[i]
            if f < 0 or f >= n_fac:
                f = 0
                best_result[i] = 0
            load[f] += demands[i]
        if np.any(load > capacities):
            best_result = np.zeros(n_cust, dtype=np.int64)
            best_load = np.zeros(n_fac, dtype=np.int64)
            for i in range(n_cust):
                best_load[0] += demands[i]

    return best_result.astype(int)

# EVOLVE-BLOCK-END
