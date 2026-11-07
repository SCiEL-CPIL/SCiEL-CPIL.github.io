# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility respecting capacities.
    Minimizes open_cost (for open facilities) + assign_cost.
    Strategy: Diversified greedy initializations + aggressive local search
    with 1-move, 2-swap, and facility-close moves.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    start_time = time.time()
    deadline = start_time + 140.0  # safety margin for 150s kill

    def _compute_obj(result, load):
        open_facs = load > 0
        return np.sum(open_cost[open_facs]) + np.sum(assign_cost[np.arange(n_cust), result])

    def _greedy_assign(order):
        """Greedy: assign customers in given order to cheapest feasible facility."""
        fac_order = np.argsort(assign_cost, axis=1)
        load = np.zeros(n_fac, dtype=np.int64)
        result = np.full(n_cust, -1, dtype=np.int64)
        for i in order:
            for f in fac_order[i]:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break
            if result[i] == -1:
                # Fallback: any facility with space
                for f in range(n_fac):
                    if load[f] + demands[i] <= capacities[f]:
                        load[f] += demands[i]
                        result[i] = f
                        break
            if result[i] == -1:
                # Force: put in facility with most remaining capacity
                remaining = capacities - load
                f = int(np.argmax(remaining))
                load[f] += demands[i]
                result[i] = f
        return result, load

    def _force_valid(result, load):
        """Repair any capacity violations by moving customers to feasible facilities."""
        for i in range(n_cust):
            if load[result[i]] > capacities[result[i]]:
                # Need to move customer i out
                for f in np.argsort(assign_cost[i]):
                    if f == result[i]:
                        continue
                    if load[f] + demands[i] <= capacities[f]:
                        load[result[i]] -= demands[i]
                        load[f] += demands[i]
                        result[i] = f
                        break
        return result, load

    def _local_search(result, load):
        """Aggressive local search: 1-move, 2-swap, and facility-close moves."""
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # 1-move: move single customer to better facility
            for i in range(n_cust):
                current_f = result[i]
                current_cost = assign_cost[i, current_f]
                # Try moving to a better facility
                for f in np.argsort(assign_cost[i]):
                    if f == current_f:
                        continue
                    if load[f] + demands[i] <= capacities[f]:
                        delta_assign = assign_cost[i, f] - current_cost
                        delta_open = 0.0
                        # If current facility becomes empty, save open_cost
                        if load[current_f] == demands[i]:
                            delta_open -= open_cost[current_f]
                        # If new facility was empty, pay open_cost
                        if load[f] == 0:
                            delta_open += open_cost[f]
                        if delta_assign + delta_open < -1e-9:
                            load[current_f] -= demands[i]
                            load[f] += demands[i]
                            result[i] = f
                            improved = True
                            break
            if time.time() >= deadline - 5:
                break
            # 2-swap: swap two customers between facilities
            for i in range(n_cust):
                for j in range(i + 1, n_cust):
                    if time.time() >= deadline - 5:
                        break
                    fi = result[i]
                    fj = result[j]
                    if fi == fj:
                        continue
                    # Check capacity after swap
                    if load[fj] - demands[j] + demands[i] <= capacities[fj] and \
                       load[fi] - demands[i] + demands[j] <= capacities[fi]:
                        delta = (assign_cost[i, fj] + assign_cost[j, fi]) - \
                                (assign_cost[i, fi] + assign_cost[j, fj])
                        delta_open = 0.0
                        # Check if any facility becomes empty
                        if load[fi] == demands[i] + demands[j]:
                            # Both customers leaving fi? No, i leaves fi, j leaves fj
                            # After swap: fi loses i, gains j. If load[fi] - demands[i] + demands[j] == 0, fi becomes empty
                            pass
                        # More careful: compute new loads
                        new_load_fi = load[fi] - demands[i] + demands[j]
                        new_load_fj = load[fj] - demands[j] + demands[i]
                        if new_load_fi == 0 and load[fi] > 0:
                            delta_open -= open_cost[fi]
                        if new_load_fj == 0 and load[fj] > 0:
                            delta_open -= open_cost[fj]
                        if delta + delta_open < -1e-9:
                            load[fi] = new_load_fi
                            load[fj] = new_load_fj
                            result[i] = fj
                            result[j] = fi
                            improved = True
                            break
                if time.time() >= deadline - 5:
                    break
            if time.time() >= deadline - 5:
                break
            # Facility-close move: reassign all customers from a low-utilization facility
            for f in range(n_fac):
                if time.time() >= deadline - 5:
                    break
                if load[f] == 0:
                    continue
                # Try closing facility f: reassign all its customers
                customers_f = np.where(result == f)[0]
                if len(customers_f) == 0:
                    continue
                # Tentatively remove all from f
                new_load = load.copy()
                new_load[f] = 0
                new_result = result.copy()
                feasible = True
                for i in customers_f:
                    # Find best facility for i (not f)
                    best_f2 = -1
                    best_cost = np.inf
                    for f2 in np.argsort(assign_cost[i]):
                        if f2 == f:
                            continue
                        if new_load[f2] + demands[i] <= capacities[f2]:
                            if assign_cost[i, f2] < best_cost:
                                best_cost = assign_cost[i, f2]
                                best_f2 = f2
                            break
                    if best_f2 == -1:
                        feasible = False
                        break
                    new_load[best_f2] += demands[i]
                    new_result[i] = best_f2
                if not feasible:
                    continue
                # Compute delta
                old_open = np.sum(open_cost[load > 0])
                new_open = np.sum(open_cost[new_load > 0])
                old_assign = np.sum(assign_cost[np.arange(n_cust), result])
                new_assign = np.sum(assign_cost[np.arange(n_cust), new_result])
                delta = (new_open + new_assign) - (old_open + old_assign)
                if delta < -1e-9:
                    load[:] = new_load
                    result[:] = new_result
                    improved = True
        return result, load

    # Generate multiple diversified initializations
    base_order = np.argsort(-demands)  # largest first
    init_orders = [base_order]
    # Reverse
    init_orders.append(base_order[::-1].copy())
    # Random shuffles
    rng = np.random.default_rng(42)
    for _ in range(5):
        shuffled = base_order.copy()
        rng.shuffle(shuffled)
        init_orders.append(shuffled)

    best_result = None
    best_obj = np.inf
    best_load = None

    for order in init_orders:
        if time.time() >= deadline - 10:
            break
        result, load = _greedy_assign(order)
        result, load = _force_valid(result, load)
        obj = _compute_obj(result, load)
        if obj < best_obj:
            best_obj = obj
            best_result = result.copy()
            best_load = load.copy()

    if best_result is None:
        # Emergency fallback
        result, load = _greedy_assign(base_order)
        result, load = _force_valid(result, load)
        return result.astype(int)

    # Local search on the best
    best_result, best_load = _local_search(best_result, best_load)

    # Final validity check
    final_load = np.zeros(n_fac, dtype=np.int64)
    for i in range(n_cust):
        f = best_result[i]
        if f < 0 or f >= n_fac:
            f = 0
            best_result[i] = 0
        final_load[f] += demands[i]
    if np.any(final_load > capacities):
        best_result, best_load = _force_valid(best_result, final_load)

    return best_result.astype(int)

# EVOLVE-BLOCK-END
