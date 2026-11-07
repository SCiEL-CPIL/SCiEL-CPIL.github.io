"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _compute_cost(result, demands, capacities, open_cost, assign_cost):
    """Compute the objective value for a given assignment. Returns inf if invalid."""
    n_fac = len(capacities)
    n_cust = len(demands)
    if len(result) != n_cust:
        return float('inf')
    load = np.zeros(n_fac)
    for i, f in enumerate(result):
        if f < 0 or f >= n_fac:
            return float('inf')
        load[f] += demands[i]
    if np.any(load > capacities + 1e-9):
        return float('inf')
    open_facilities = np.unique(result)
    cost = np.sum(open_cost[open_facilities]) + np.sum(assign_cost[np.arange(n_cust), result])
    return cost


def _greedy_assign(demands, capacities, open_cost, assign_cost):
    """Largest demand first, cheapest feasible facility."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    order = np.argsort(-demands)
    for i in order:
        feasible = []
        for f in np.argsort(assign_cost[i]):
            if load[f] + demands[i] <= capacities[f]:
                feasible.append(f)
        if not feasible:
            remaining = capacities - load
            f = np.argmax(remaining)
            if remaining[f] >= demands[i]:
                load[f] += demands[i]
                result[i] = f
            else:
                result[i] = 0
        else:
            f = feasible[0]
            load[f] += demands[i]
            result[i] = f
    return result


def _fast_local_search(result, demands, capacities, open_cost, assign_cost, time_limit=130.0):
    """
    Fast local search with incremental cost updates.
    Tries single reassignments and pairwise swaps.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    current = result.astype(int).copy()
    load = np.zeros(n_fac)
    for i, f in enumerate(current):
        load[f] += demands[i]

    # Precompute: for each facility, track how many customers are assigned
    fac_count = np.zeros(n_fac, dtype=int)
    for f in current:
        fac_count[f] += 1

    def get_open_cost():
        return np.sum(open_cost[fac_count > 0])

    def get_assign_cost():
        return np.sum(assign_cost[np.arange(n_cust), current])

    current_cost = get_open_cost() + get_assign_cost()
    best_cost = current_cost
    best_result = current.copy()
    start = time.time()

    no_improve_passes = 0
    max_no_improve = 3

    while time.time() - start < time_limit:
        improved = False
        # Try single reassignments
        for i in range(n_cust):
            if time.time() - start > time_limit:
                break
            cur_f = current[i]
            d_i = demands[i]
            # Try moving to each other facility
            for new_f in range(n_fac):
                if new_f == cur_f:
                    continue
                if load[new_f] + d_i > capacities[new_f]:
                    continue

                # Delta cost calculation
                delta = 0.0
                # Assignment cost change
                delta += assign_cost[i, new_f] - assign_cost[i, cur_f]

                # Open cost change
                # After move: cur_f count decreases by 1, new_f count increases by 1
                new_cur_count = fac_count[cur_f] - 1
                new_new_count = fac_count[new_f] + 1

                if new_cur_count == 0 and fac_count[cur_f] > 0:
                    # cur_f becomes closed
                    delta -= open_cost[cur_f]
                if new_new_count == 1 and fac_count[new_f] == 0:
                    # new_f becomes open
                    delta += open_cost[new_f]

                if delta < -1e-9:
                    # Apply move
                    load[cur_f] -= d_i
                    load[new_f] += d_i
                    fac_count[cur_f] -= 1
                    fac_count[new_f] += 1
                    current[i] = new_f
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_result = current.copy()
                    break  # restart inner loop for this customer

        # Try pairwise swaps
        if time.time() - start < time_limit:
            for i in range(n_cust):
                if time.time() - start > time_limit:
                    break
                for j in range(i + 1, n_cust):
                    if time.time() - start > time_limit:
                        break
                    fi = current[i]
                    fj = current[j]
                    if fi == fj:
                        continue
                    di = demands[i]
                    dj = demands[j]

                    # Check capacity: after swap, fi gets dj, fj gets di
                    if load[fi] - di + dj > capacities[fi]:
                        continue
                    if load[fj] - dj + di > capacities[fj]:
                        continue

                    # Delta cost
                    delta = 0.0
                    # Assignment cost
                    delta += assign_cost[i, fj] + assign_cost[j, fi] - assign_cost[i, fi] - assign_cost[j, fj]

                    # Open cost change
                    new_fi_count = fac_count[fi] - 1 + 1  # fi loses i, gains j -> same count if fi != fj
                    new_fj_count = fac_count[fj] - 1 + 1  # fj loses j, gains i -> same count

                    # Actually counts don't change for a swap between two different facilities
                    # fac_count[fi] stays the same, fac_count[fj] stays the same
                    # So no open cost change
                    # Wait: fi loses customer i and gains customer j, so count stays same
                    # fj loses customer j and gains customer i, so count stays same
                    # So delta for open cost is 0

                    if delta < -1e-9:
                        # Apply swap
                        load[fi] = load[fi] - di + dj
                        load[fj] = load[fj] - dj + di
                        current[i] = fj
                        current[j] = fi
                        current_cost += delta
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_result = current.copy()

        if not improved:
            no_improve_passes += 1
            if no_improve_passes >= max_no_improve:
                break
        else:
            no_improve_passes = 0

    return best_result


def _try_multiple_greedy(demands, capacities, open_cost, assign_cost, n_restarts=5):
    """Try multiple greedy orderings and pick the best."""
    n_cust = len(demands)
    best_result = None
    best_cost = float('inf')

    # Standard: largest demand first
    result = _greedy_assign(demands, capacities, open_cost, assign_cost)
    cost = _compute_cost(result, demands, capacities, open_cost, assign_cost)
    if cost < best_cost:
        best_cost = cost
        best_result = result.copy()

    # Smallest demand first
    n_fac = len(capacities)
    result2 = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    order2 = np.argsort(demands)
    for i in order2:
        for f in np.argsort(assign_cost[i]):
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result2[i] = f
                break
        else:
            remaining = capacities - load
            f = np.argmax(remaining)
            if remaining[f] >= demands[i]:
                load[f] += demands[i]
                result2[i] = f
            else:
                result2[i] = 0
    cost2 = _compute_cost(result2, demands, capacities, open_cost, assign_cost)
    if cost2 < best_cost:
        best_cost = cost2
        best_result = result2.copy()

    # Cheapest assignment cost first (sum of sorted assign costs per customer)
    result3 = np.zeros(n_cust, dtype=int)
    load3 = np.zeros(n_fac)
    customer_order = np.argsort(np.min(assign_cost, axis=1))
    for i in customer_order:
        for f in np.argsort(assign_cost[i]):
            if load3[f] + demands[i] <= capacities[f]:
                load3[f] += demands[i]
                result3[i] = f
                break
        else:
            remaining = capacities - load3
            f = np.argmax(remaining)
            if remaining[f] >= demands[i]:
                load3[f] += demands[i]
                result3[i] = f
            else:
                result3[i] = 0
    cost3 = _compute_cost(result3, demands, capacities, open_cost, assign_cost)
    if cost3 < best_cost:
        best_cost = cost3
        best_result = result3.copy()

    return best_result


def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin before 150s kill

    n_cust = len(demands)
    n_fac = len(capacities)

    # Try multiple greedy initializations
    result = _try_multiple_greedy(demands, capacities, open_cost, assign_cost)

    # Fast local search with time limit
    remaining_time = deadline - time.time()
    if remaining_time > 2:
        result = _fast_local_search(result, demands, capacities, open_cost, assign_cost,
                                    time_limit=remaining_time - 2)

    # Final validation
    final_cost = _compute_cost(result, demands, capacities, open_cost, assign_cost)
    if final_cost == float('inf'):
        # Fallback to simple greedy
        result = _greedy_assign(demands, capacities, open_cost, assign_cost)

    return result
# EVOLVE-BLOCK-END
