# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using a portfolio of greedy seeds,
    and best-improvement local search.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    n_cust = len(demands)
    n_fac = len(capacities)

    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    def _total_cost(result):
        load = np.zeros(n_fac)
        for c in range(n_cust):
            load[result[c]] += demands[c]
        used = load > 0
        open_sum = open_cost[used].sum()
        assign_sum = np.sum(assign_cost[np.arange(n_cust), result])
        return open_sum + assign_sum

    def _is_feasible(result, load):
        return np.all(load <= capacities + 1e-6)

    def _greedy_seed(strategy='eff'):
        """Generate initial assignment using various greedy strategies."""
        result = np.full(n_cust, -1, dtype=int)
        load = np.zeros(n_fac, dtype=float)

        amortized = open_cost / np.maximum(capacities, 1.0)

        if strategy == 'eff':
            eff_cost = assign_cost + np.outer(demands, amortized)
            order = np.argsort(-demands)
            cost_matrix = eff_cost
        elif strategy == 'eff_desc':
            eff_cost = assign_cost + np.outer(demands, amortized)
            order = np.argsort(-demands)
            cost_matrix = eff_cost
        elif strategy == 'raw_desc':
            order = np.argsort(-demands)
            cost_matrix = assign_cost
        elif strategy == 'raw_asc':
            order = np.argsort(demands)
            cost_matrix = assign_cost
        elif strategy == 'random':
            order = np.random.permutation(n_cust)
            cost_matrix = assign_cost
        elif strategy == 'cheapest':
            min_costs = np.min(assign_cost, axis=1)
            order = np.argsort(min_costs)
            cost_matrix = assign_cost
        else:
            order = np.argsort(-demands)
            cost_matrix = assign_cost

        for i in order:
            fac_order = np.argsort(cost_matrix[i])
            assigned = False
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break
            if not assigned:
                for f in np.argsort(assign_cost[i]):
                    if load[f] + demands[i] <= capacities[f]:
                        load[f] += demands[i]
                        result[i] = f
                        break
            if result[i] < 0:
                remaining = capacities - load
                f = int(np.argmax(remaining))
                result[i] = f
                load[f] += demands[i]

        return result, load

    def _local_search(result, load, deadline):
        """Best-improvement local search with Relocate and Swap moves."""
        result = result.copy()
        load = load.copy()
        count = np.zeros(n_fac, dtype=int)
        for c in range(n_cust):
            count[result[c]] += 1

        current_cost = _total_cost(result)

        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False

            # --- Phase 1: Best Relocate ---
            best_delta = 0.0
            best_c = -1
            best_f = -1

            for c in range(n_cust):
                if time.time() > deadline - 5.0:
                    break
                old_f = result[c]
                d_c = demands[c]
                base_cost = assign_cost[c, old_f]

                for f in range(n_fac):
                    if f == old_f:
                        continue
                    if load[f] + d_c > capacities[f]:
                        continue

                    delta = assign_cost[c, f] - base_cost

                    # Open cost changes
                    if count[old_f] == 1:
                        delta -= open_cost[old_f]
                    if count[f] == 0:
                        delta += open_cost[f]

                    if delta < best_delta:
                        best_delta = delta
                        best_c = c
                        best_f = f

            if best_delta < -1e-9 and best_c != -1:
                c = best_c
                f = best_f
                old_f = result[c]
                result[c] = f
                load[old_f] -= demands[c]
                load[f] += demands[c]
                count[old_f] -= 1
                count[f] += 1
                current_cost += best_delta
                improved = True
                continue

            # --- Phase 2: Best Swap ---
            best_swap_delta = 0.0
            best_c1 = -1
            best_c2 = -1

            for c1 in range(n_cust):
                if time.time() > deadline - 5.0:
                    break
                f1 = result[c1]
                d1 = demands[c1]
                ac1_f1 = assign_cost[c1, f1]

                for c2 in range(c1 + 1, n_cust):
                    f2 = result[c2]
                    if f1 == f2:
                        continue
                    d2 = demands[c2]

                    new_load_f1 = load[f1] - d1 + d2
                    new_load_f2 = load[f2] - d2 + d1

                    if new_load_f1 > capacities[f1] + 1e-6 or new_load_f2 > capacities[f2] + 1e-6:
                        continue

                    delta = (assign_cost[c1, f2] + assign_cost[c2, f1]) - \
                            (ac1_f1 + assign_cost[c2, f2])

                    if delta < best_swap_delta:
                        best_swap_delta = delta
                        best_c1 = c1
                        best_c2 = c2

            if best_swap_delta < -1e-9 and best_c1 != -1:
                c1, c2 = best_c1, best_c2
                f1, f2 = result[c1], result[c2]
                result[c1], result[c2] = f2, f1
                load[f1] -= demands[c1]
                load[f1] += demands[c2]
                load[f2] -= demands[c2]
                load[f2] += demands[c1]
                current_cost += best_swap_delta
                improved = True
                continue

        return result, current_cost

    # Portfolio of seeds
    strategies = ['eff', 'eff_desc', 'raw_desc', 'raw_asc', 'cheapest', 'random']
    best_cost = float('inf')
    best_result = None
    best_load = None

    # Phase 1: Quick evaluation of seeds
    for strategy in strategies:
        if time.time() > deadline - 60.0:
            break

        result, load = _greedy_seed(strategy)
        current_cost = _total_cost(result)

        # Quick local search (limit to 5s per seed)
        seed_deadline = time.time() + 5.0
        result, current_cost = _local_search(result, load, seed_deadline)

        if current_cost < best_cost:
            best_cost = current_cost
            best_result = result.copy()
            best_load = load.copy()

    if best_result is None:
        best_result, best_load = _greedy_seed('eff')
        best_cost = _total_cost(best_result)

    # Phase 2: Long refinement on the best solution
    if time.time() < deadline - 5.0:
        refined_result, refined_cost = _local_search(best_result, best_load, deadline)
        if refined_cost < best_cost:
            best_result = refined_result
            best_cost = refined_cost

    return best_result

# EVOLVE-BLOCK-END
