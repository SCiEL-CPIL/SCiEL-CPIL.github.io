"""Capacitated facility location: greedy seed + local-search refinement with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _objective(assignment, demands, capacities, open_cost, assign_cost):
    """Compute total cost: open cost for used facilities + assignment cost."""
    n_fac = len(capacities)
    used = np.zeros(n_fac, dtype=bool)
    for c, f in enumerate(assignment):
        if f >= 0 and f < n_fac:
            used[f] = True
    open_total = open_cost[used].sum()
    # Check capacity
    load = np.zeros(n_fac, dtype=float)
    for c, f in enumerate(assignment):
        if 0 <= f < n_fac:
            load[f] += demands[c]
    if np.any(load > capacities + 1e-9):
        return np.inf
    assign_total = sum(assign_cost[c, assignment[c]] for c in range(len(assignment)))
    return open_total + assign_total


def _greedy_seed(demands, capacities, open_cost, assign_cost):
    """Greedy: largest demand first, cheapest feasible facility."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac, dtype=float)

    # Process in order of decreasing demand; break ties by higher total assign cost (more constrained)
    total_cost = assign_cost.sum(axis=1)
    order = np.lexsort((total_cost, -demands))

    # Pre-sort facilities by assign cost for each customer
    sorted_fac = np.argsort(assign_cost, axis=1)

    for i in order:
        placed = False
        for f in sorted_fac[i]:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                placed = True
                break
        if not placed:
            # Fallback: assign to facility with most remaining capacity
            remaining = capacities - load
            f = np.argmax(remaining)
            result[i] = f
            load[f] += demands[i]

    return result


def _local_search(assignment, demands, capacities, open_cost, assign_cost, deadline):
    """Local search: try reassigning customers to reduce cost while respecting capacity."""
    n_cust = len(assignment)
    n_fac = len(capacities)
    best = assignment.copy()
    best_cost = _objective(best, demands, capacities, open_cost, assign_cost)
    if np.isinf(best_cost):
        return best, best_cost

    load = np.zeros(n_fac, dtype=float)
    for c, f in enumerate(best):
        if 0 <= f < n_fac:
            load[f] += demands[c]

    # Pre-sort facilities by assign cost for each customer
    sorted_fac = np.argsort(assign_cost, axis=1)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try each customer in order of increasing demand (easier to reassign)
        order = np.argsort(demands)
        for i in order:
            if time.time() >= deadline:
                break
            current_f = best[i]
            current_cost = assign_cost[i, current_f]
            # Try all facilities in order of assign cost
            for f in sorted_fac[i]:
                if f == current_f:
                    continue
                # Check if moving is beneficial
                new_cost = assign_cost[i, f]
                if new_cost >= current_cost:
                    break  # Since sorted, no further improvement possible for this customer

                # Check capacity at f
                if load[f] + demands[i] > capacities[f]:
                    continue

                # Calculate delta: change in assign cost + change in open cost
                delta_assign = new_cost - current_cost
                # Open cost change: if f was not used before, we add open_cost[f]
                # If current_f becomes empty, we remove open_cost[current_f]
                delta_open = 0.0
                if load[f] == 0.0:
                    delta_open += open_cost[f]
                if load[current_f] - demands[i] <= 1e-9:
                    delta_open -= open_cost[current_f]

                total_delta = delta_assign + delta_open
                if total_delta < -1e-9:
                    # Accept move
                    load[current_f] -= demands[i]
                    load[f] += demands[i]
                    best[i] = f
                    best_cost += total_delta
                    improved = True
                    break  # Move to next customer

    return best, best_cost


def assign(demands, capacities, open_cost, assign_cost):
    """Assign each customer to a facility minimizing total cost."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    # Greedy seed
    result = _greedy_seed(demands, capacities, open_cost, assign_cost)

    # Local search refinement
    result, cost = _local_search(result, demands, capacities, open_cost, assign_cost, deadline)

    return result.astype(int)
# EVOLVE-BLOCK-END
