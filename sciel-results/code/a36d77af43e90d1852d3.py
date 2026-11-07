"""Capacitated facility location solver. EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _total_cost(result, assign_cost, open_cost):
    n_cust = len(result)
    n_fac = len(open_cost)
    used = np.zeros(n_fac, dtype=bool)
    used[result] = True
    return float(np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(n_cust), result]))


def _greedy(demands, capacities, assign_cost, order):
    """Assign in given order to cheapest feasible facility; feasibility guaranteed."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac)
    for i in order:
        d = demands[i]
        rem = capacities - load
        feas = rem >= d - 1e-9
        if np.any(feas):
            cands = np.where(feas)[0]
            f = cands[np.argmin(assign_cost[i, cands])]
        else:
            f = int(np.argmax(rem))
        load[f] += d
        result[i] = f
    return result.astype(int)


def _local_search(result, demands, capacities, assign_cost, open_cost, deadline):
    """Vectorized first-improvement 1-move local search."""
    n_cust = len(result)
    n_fac = len(capacities)
    result = result.copy()
    load = np.zeros(n_fac)
    cnt = np.zeros(n_fac, dtype=int)
    for i in range(n_cust):
        load[result[i]] += demands[i]
        cnt[result[i]] += 1
    best_cost = _total_cost(result, assign_cost, open_cost)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n_cust):
            if time.time() > deadline:
                break
            cur = result[i]
            d = demands[i]
            rem = capacities - load
            feas = (rem >= d - 1e-9) & (np.arange(n_fac) != cur)
            if not np.any(feas):
                continue
            cands = np.where(feas)[0]
            delta = assign_cost[i, cands] - assign_cost[i, cur]
            delta[cnt[cur] == 1] -= open_cost[cur]
            delta[cnt[cands] == 0] += open_cost[cands]
            k = int(np.argmin(delta))
            if delta[k] < -1e-9:
                f = cands[k]
                load[cur] -= d
                load[f] += d
                cnt[cur] -= 1
                cnt[f] += 1
                result[i] = f
                best_cost += delta[k]
                improved = True
    return result, best_cost


def _swap_pass(result, demands, capacities, assign_cost, open_cost, deadline):
    """2-opt swap between customers at different facilities."""
    n_cust = len(result)
    n_fac = len(capacities)
    result = result.copy()
    load = np.zeros(n_fac)
    cnt = np.zeros(n_fac, dtype=int)
    for i in range(n_cust):
        load[result[i]] += demands[i]
        cnt[result[i]] += 1
    best_cost = _total_cost(result, assign_cost, open_cost)
    groups = {f: [] for f in range(n_fac)}
    for i in range(n_cust):
        groups[result[i]].append(i)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for f1 in range(n_fac):
            if time.time() > deadline:
                break
            g1 = groups[f1]
            if not g1:
                continue
            for f2 in range(f1 + 1, n_fac):
                if time.time() > deadline:
                    break
                g2 = groups[f2]
                if not g2:
                    continue
                L1, L2 = load[f1], load[f2]
                cap1, cap2 = capacities[f1], capacities[f2]
                for i in list(g1):
                    if time.time() > deadline:
                        break
                    di = demands[i]
                    for j in list(g2):
                        dj = demands[j]
                        if L1 - di + dj > cap1 + 1e-6:
                            continue
                        if L2 - dj + di > cap2 + 1e-6:
                            continue
                        delta = (assign_cost[i, f2] + assign_cost[j, f1]) - \
                                (assign_cost[i, f1] + assign_cost[j, f2])
                        if len(g1) == 1:
                            delta -= open_cost[f1]
                        if len(g2) == 1:
                            delta -= open_cost[f2]
                        if delta < -1e-9:
                            load[f1] += dj - di
                            load[f2] += di - dj
                            result[i] = f2
                            result[j] = f1
                            best_cost += delta
                            improved = True
                            g1.remove(i)
                            g2.remove(j)
                            g2.append(i)
                            g1.append(j)
                            break
    return result, best_cost


def assign(demands, capacities, open_cost, assign_cost):
    start = time.time()
    deadline = start + 143.0
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    n_cust = len(demands)
    n_fac = len(capacities)

    order_largest = np.argsort(-demands)
    order_cheapest = np.lexsort((np.argmax(assign_cost, axis=1), np.arange(n_cust)))

    # Diverse initializations
    init_a = _greedy(demands, capacities, assign_cost, order_largest)
    init_b = _greedy(demands, capacities, assign_cost, order_cheapest)

    best_result = init_a
    best_cost = _total_cost(init_a, assign_cost, open_cost)
    if _total_cost(init_b, assign_cost, open_cost) < best_cost:
        best_result = init_b
        best_cost = _total_cost(init_b, assign_cost, open_cost)

    # Refine both starting points, keep the better
    for init in (init_a, init_b):
        if time.time() > deadline - 3.0:
            break
        r, c = _local_search(init, demands, capacities, assign_cost, open_cost, deadline - 3.0)
        r, c = _swap_pass(r, demands, capacities, assign_cost, open_cost, deadline - 3.0)
        r, c = _local_search(r, demands, capacities, assign_cost, open_cost, deadline - 3.0)
        if c < best_cost:
            best_cost = c
            best_result = r

    # Final polish
    if time.time() < deadline - 2.0:
        r, c = _local_search(best_result, demands, capacities, assign_cost, open_cost, deadline - 2.0)
        if c < best_cost:
            best_cost = c
            best_result = r

    # Feasibility guard
    load = np.zeros(n_fac)
    for i in range(n_cust):
        f = best_result[i]
        if f < 0 or f >= n_fac:
            f = 0
        best_result[i] = f
        load[f] += demands[i]
    if np.any(load > capacities + 1e-6):
        best_result = _greedy(demands, capacities, assign_cost, order_largest)

    return best_result.astype(int)
# EVOLVE-BLOCK-END
