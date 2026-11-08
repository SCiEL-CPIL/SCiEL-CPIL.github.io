"""Capacitated facility location: cost-aware greedy seed + local search with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _total_cost(result, demands, capacities, open_cost, assign_cost):
    """Compute total cost: open costs + assignment costs."""
    n_fac = len(open_cost)
    load = np.zeros(n_fac)
    for c, f in enumerate(result):
        load[f] += demands[c]
    open_sum = open_cost[np.any(load > 0, axis=0) if False else (load > 0)]
    # Actually, load is 1D, so:
    used = load > 0
    open_sum = open_cost[used].sum()
    assign_sum = sum(assign_cost[c, result[c]] for c in range(len(result)))
    return open_sum + assign_sum


def _compute_load(result, demands, n_fac):
    load = np.zeros(n_fac)
    for c, f in enumerate(result):
        load[f] += demands[c]
    return load


def _greedy_seed(demands, capacities, open_cost, assign_cost):
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac, dtype=float)

    # Effective cost: assignment cost + amortized open cost
    # Amortized open cost per unit demand for facility f: open_cost[f] / max(capacity[f], 1)
    amortized = open_cost / np.maximum(capacities, 1.0)
    # Effective cost matrix: assign_cost[i,f] + amortized[f] * demands[i]
    # This approximates the marginal cost of assigning customer i to facility f
    eff_cost = assign_cost + np.outer(demands, amortized)

    # Sort customers: larger demand first, tie-break by higher total effective cost (more constrained)
    total_eff = eff_cost.sum(axis=1)
    order = np.lexsort((total_eff, -demands))

    for i in order:
        # Sort facilities by effective cost
        fac_order = np.argsort(eff_cost[i])
        for f in fac_order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
        else:
            # Fallback: cheapest feasible by raw assign_cost
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break

    return result


def _local_search(result, demands, capacities, open_cost, assign_cost, deadline):
    """Iterative local search with reassign and swap moves."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = result.copy()
    load = _compute_load(result, demands, n_fac)
    current_cost = _total_cost(result, demands, capacities, open_cost, assign_cost)

    # Precompute: for each customer, sorted list of facilities by assignment cost
    fac_order = np.argsort(assign_cost, axis=1)

    improved = True
    while improved and time.time() < deadline:
        improved = False

        # --- Reassign moves ---
        # Try moving each customer to a different facility
        for c in range(n_cust):
            if time.time() >= deadline:
                break
            old_f = result[c]
            for f in fac_order[c]:
                if f == old_f:
                    continue
                if load[f] + demands[c] <= capacities[f]:
                    # Compute delta
                    # Remove c from old_f: if old_f becomes empty, we save open_cost[old_f]
                    new_load_old = load[old_f] - demands[c]
                    delta = assign_cost[c, f] - assign_cost[c, old_f]
                    if new_load_old <= 0:
                        delta -= open_cost[old_f]  # close old facility
                    # If f was empty before, we pay open_cost[f]
                    if load[f] <= 0:
                        delta += open_cost[f]
                    if delta < -1e-9:
                        # Apply move
                        load[old_f] -= demands[c]
                        load[f] += demands[c]
                        result[c] = f
                        current_cost += delta
                        improved = True
                        break  # move this customer at most once per pass

        # --- Swap moves ---
        # Try swapping two customers between two different facilities
        for c1 in range(n_cust):
            if time.time() >= deadline:
                break
            f1 = result[c1]
            for c2 in range(c1 + 1, n_cust):
                if time.time() >= deadline:
                    break
                f2 = result[c2]
                if f1 == f2:
                    continue
                # Check capacity after swap
                if load[f1] - demands[c1] + demands[c2] > capacities[f1]:
                    continue
                if load[f2] - demands[c2] + demands[c1] > capacities[f2]:
                    continue
                # Compute delta
                delta = (assign_cost[c1, f2] + assign_cost[c2, f1]) - \
                        (assign_cost[c1, f1] + assign_cost[c2, f2])
                # Open cost changes
                new_load_f1 = load[f1] - demands[c1] + demands[c2]
                new_load_f2 = load[f2] - demands[c2] + demands[c1]
                old_f1_used = load[f1] > 0
                old_f2_used = load[f2] > 0
                new_f1_used = new_load_f1 > 0
                new_f2_used = new_load_f2 > 0
                if old_f1_used and not new_f1_used:
                    delta -= open_cost[f1]
                elif not old_f1_used and new_f1_used:
                    delta += open_cost[f1]
                if old_f2_used and not new_f2_used:
                    delta -= open_cost[f2]
                elif not old_f2_used and new_f2_used:
                    delta += open_cost[f2]
                if delta < -1e-9:
                    # Apply swap
                    load[f1] = new_load_f1
                    load[f2] = new_load_f2
                    result[c1] = f2
                    result[c2] = f1
                    current_cost += delta
                    improved = True
                    break  # c1 moved, restart c2 loop implicitly by breaking

    return result, current_cost


def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    n_cust = len(demands)
    n_fac = len(capacities)

    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    best_result = None
    best_cost = float('inf')

    # Run multiple greedy seeds with different orderings, then local search
    # Strategy 1: standard greedy (largest demand first, effective cost)
    result1 = _greedy_seed(demands, capacities, open_cost, assign_cost)
    if np.all(result1 >= 0):
        cost1 = _total_cost(result1, demands, capacities, open_cost, assign_cost)
        if cost1 < best_cost:
            best_cost = cost1
            best_result = result1.copy()

    # Local search on best seed
    if time.time() < deadline:
        result_ls, cost_ls = _local_search(best_result, demands, capacities, open_cost, assign_cost, deadline)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_result = result_ls

    # Strategy 2: greedy with raw assign_cost (no open cost amortization)
    if time.time() < deadline - 5:
        result2 = _greedy_seed(demands, capacities, open_cost * 0.0, assign_cost)
        if np.all(result2 >= 0):
            cost2 = _total_cost(result2, demands, capacities, open_cost, assign_cost)
            if cost2 < best_cost:
                best_cost = cost2
                best_result = result2.copy()
                # Local search on this
                if time.time() < deadline:
                    result_ls2, cost_ls2 = _local_search(best_result, demands, capacities, open_cost, assign_cost, deadline)
                    if cost_ls2 < best_cost:
                        best_cost = cost_ls2
                        best_result = result_ls2

    # Strategy 3: greedy with higher open cost weight (encourage fewer facilities)
    if time.time() < deadline - 5:
        result3 = _greedy_seed(demands, capacities, open_cost * 2.0, assign_cost)
        if np.all(result3 >= 0):
            cost3 = _total_cost(result3, demands, capacities, open_cost, assign_cost)
            if cost3 < best_cost:
                best_cost = cost3
                best_result = result3.copy()
                if time.time() < deadline:
                    result_ls3, cost_ls3 = _local_search(best_result, demands, capacities, open_cost, assign_cost, deadline)
                    if cost_ls3 < best_cost:
                        best_cost = cost_ls3
                        best_result = result_ls3

    # Ensure all customers are assigned
    if best_result is None or np.any(best_result < 0):
        # Fallback: simple greedy
        best_result = _greedy_seed(demands, capacities, open_cost, assign_cost)

    return best_result.astype(int)
# EVOLVE-BLOCK-END
