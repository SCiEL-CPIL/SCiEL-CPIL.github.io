import numpy as np
import time


def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    n_customers = len(demands)
    n_facilities = len(capacities)

    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    best_result = None
    best_score = float('inf')

    def evaluate(result):
        load = np.zeros(n_facilities)
        for i, f in enumerate(result):
            if f < 0 or f >= n_facilities:
                return float('inf')
            load[f] += demands[i]
        if np.any(load > capacities + 1e-9):
            return float('inf')

        open_mask = np.zeros(n_facilities, dtype=bool)
        for f in result:
            open_mask[f] = True

        open_sum = np.sum(open_cost[open_mask])
        assign_sum = np.sum(assign_cost[np.arange(n_customers), result])
        return open_sum + assign_sum

    def greedy_init(randomize=False, seed=None):
        if seed is not None:
            np.random.seed(seed)

        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        is_open = np.zeros(n_facilities, dtype=bool)

        # Sort customers by demand descending
        order = np.argsort(-demands)

        for i in order:
            d = demands[i]
            # Feasible facilities
            feasible = load + d <= capacities

            if not np.any(feasible):
                f = np.argmin(assign_cost[i])
                result[i] = f
                load[f] += d
                is_open[f] = True
                continue

            # Calculate marginal costs
            marginal_costs = assign_cost[i].copy()
            closed_mask = ~is_open
            marginal_costs[closed_mask] += open_cost[closed_mask]

            # Mask infeasible
            marginal_costs[~feasible] = np.inf

            if randomize:
                # Pick from top 3
                sorted_indices = np.argsort(marginal_costs)
                top_k = sorted_indices[:min(3, n_facilities)]
                f = top_k[np.random.randint(len(top_k))]
            else:
                f = np.argmin(marginal_costs)

            result[i] = f
            load[f] += d
            is_open[f] = True

        return result, load, is_open

    def local_search(result, load, is_open):
        improved = True
        while improved:
            improved = False
            # Iterate customers in fixed order
            for i in np.arange(n_customers):
                f_old = result[i]
                d = demands[i]

                best_delta = 1e-9
                best_f_new = -1

                for f_new in range(n_facilities):
                    if f_new == f_old:
                        continue

                    # Check capacity
                    if load[f_new] + d > capacities[f_new]:
                        continue

                    # Calculate delta
                    delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]

                    delta_open = 0.0
                    # Check if old facility becomes empty
                    if load[f_old] - d == 0:
                        delta_open -= open_cost[f_old]

                    # Check if new facility was empty
                    if load[f_new] == 0:
                        if not is_open[f_new]:
                            delta_open += open_cost[f_new]

                    total_delta = delta_assign + delta_open

                    if total_delta < best_delta:
                        best_delta = total_delta
                        best_f_new = f_new

                if best_f_new != -1:
                    # Apply move
                    load[f_old] -= d
                    if load[f_old] == 0:
                        is_open[f_old] = False

                    load[best_f_new] += d
                    if load[best_f_new] == d:  # Was 0
                        is_open[best_f_new] = True

                    result[i] = best_f_new
                    improved = True

                    if time.time() > deadline:
                        return result, load, is_open

        return result, load, is_open

    def swap_search(result, load, is_open):
        improved = True
        while improved:
            improved = False
            for i in range(n_customers):
                f_i = result[i]
                d_i = demands[i]

                for j in range(i + 1, n_customers):
                    f_j = result[j]
                    d_j = demands[j]

                    if f_i == f_j:
                        continue

                    # Check capacity after swap
                    if load[f_i] - d_i + d_j > capacities[f_i]:
                        continue
                    if load[f_j] - d_j + d_i > capacities[f_j]:
                        continue

                    # Calculate delta
                    delta_assign = (assign_cost[i, f_j] + assign_cost[j, f_i]) - \
                                   (assign_cost[i, f_i] + assign_cost[j, f_j])

                    total_delta = delta_assign

                    if total_delta < -1e-9:
                        # Apply swap
                        result[i] = f_j
                        result[j] = f_i
                        load[f_i] -= d_i
                        load[f_i] += d_j
                        load[f_j] -= d_j
                        load[f_j] += d_i
                        improved = True
                        break

                if time.time() > deadline:
                    return result, load, is_open

        return result, load, is_open

    def two_move_search(result, load, is_open):
        """Try moving two customers simultaneously to escape local optima."""
        improved = True
        while improved:
            improved = False
            for i in range(n_customers):
                f_i = result[i]
                d_i = demands[i]
                for j in range(i + 1, n_customers):
                    f_j = result[j]
                    d_j = demands[j]
                    if f_i == f_j:
                        continue

                    # Try moving i to f_j and j to f_i (swap already covered)
                    # Try moving i to some f_a and j to some f_b
                    # This is expensive, so limit to a few candidates
                    # For now, just do a simple 2-opt style
                    pass
            if time.time() > deadline:
                break
        return result, load, is_open

    def facility_reopen(result, load, is_open):
        """Try to close a facility and redistribute its customers."""
        improved = True
        while improved:
            improved = False
            for f_old in range(n_facilities):
                if not is_open[f_old]:
                    continue
                # Find all customers at f_old
                customers_at = [i for i in range(n_customers) if result[i] == f_old]
                if not customers_at:
                    continue

                # Try to move all customers from f_old to other facilities
                # Check if it's beneficial
                old_open_cost = open_cost[f_old]
                old_assign_cost = sum(assign_cost[i, f_old] for i in customers_at)

                # Try to redistribute
                temp_load = load.copy()
                temp_load[f_old] = 0
                temp_is_open = is_open.copy()
                temp_is_open[f_old] = False

                new_assign_cost = 0.0
                new_open_cost = 0.0
                feasible = True

                for i in customers_at:
                    d = demands[i]
                    best_f = -1
                    best_cost = float('inf')
                    for f_new in range(n_facilities):
                        if f_new == f_old:
                            continue
                        if temp_load[f_new] + d > capacities[f_new]:
                            continue
                        cost = assign_cost[i, f_new]
                        if not temp_is_open[f_new]:
                            cost += open_cost[f_new]
                        if cost < best_cost:
                            best_cost = cost
                            best_f = f_new
                    if best_f == -1:
                        feasible = False
                        break
                    new_assign_cost += assign_cost[i, best_f]
                    if not temp_is_open[best_f]:
                        new_open_cost += open_cost[best_f]
                        temp_is_open[best_f] = True
                    temp_load[best_f] += d

                if not feasible:
                    continue

                old_total = old_open_cost + old_assign_cost
                new_total = new_open_cost + new_assign_cost

                if new_total < old_total - 1e-9:
                    # Apply the move
                    for i in customers_at:
                        d = demands[i]
                        # Find where this customer should go
                        # This is tricky because we need to recompute
                        pass
                    # For simplicity, skip applying and just note it
                    improved = True
                    break

            if time.time() > deadline:
                break
        return result, load, is_open

    # Main loop with restarts
    # 1. Deterministic greedy
    res, load, is_open = greedy_init(randomize=False)
    res, load, is_open = local_search(res, load, is_open)
    res, load, is_open = swap_search(res, load, is_open)
    score = evaluate(res)
    if score < best_score:
        best_score = score
        best_result = res.copy()

    # 2. Randomized restarts
    for k in range(50):
        if time.time() > deadline - 10:  # Leave 10s buffer
            break

        res, load, is_open = greedy_init(randomize=True, seed=k)
        res, load, is_open = local_search(res, load, is_open)
        res, load, is_open = swap_search(res, load, is_open)
        score = evaluate(res)
        if score < best_score:
            best_score = score
            best_result = res.copy()

    return best_result
