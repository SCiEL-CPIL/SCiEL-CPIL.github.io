# EVOLVE-BLOCK-START
import time
import numpy as np


def _valid(res, demands, capacities):
    if res is None:
        return False
    res = np.asarray(res)
    if res.shape[0] != len(demands):
        return False
    if not np.issubdtype(res.dtype, np.integer):
        return False
    if res.min() < 0 or res.max() >= len(capacities):
        return False
    load = np.zeros(len(capacities))
    np.add.at(load, res, demands)
    return bool(np.all(load <= capacities))


def _total_cost(res, demands, capacities, open_cost, assign_cost):
    used = np.unique(res)
    return float(np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(len(demands)), res]))


def _construct(demands, capacities, open_cost, assign_cost, rng):
    n = len(demands)
    m = len(capacities)
    total = float(demands.sum())

    # Facility-opening plan (k-means-like), balancing open cost vs assignment cost.
    centers = [int(rng.integers(0, m))]
    for _ in range(3000):
        if len(centers) >= m:
            break
        # marginal cost of opening a new facility for the closest customer
        d = assign_cost[:, centers].min(axis=1)
        dmin = d.min()
        gain = dmin - open_cost.min()
        if gain <= 1e-9:
            break
        centers.append(int(np.argmin(open_cost + d)))

    # Assign each customer to its nearest open facility.
    labels = assign_cost[:, centers].argmin(axis=1)

    # Repair capacity violations by moving customers to the cheapest feasible facility.
    cap = np.asarray(capacities, dtype=float).copy()
    load = np.zeros(m)
    for c in range(n):
        load[labels[c]] += demands[c]
    for c in range(n):
        f = labels[c]
        if load[f] <= cap[f]:
            continue
        for cand in np.argsort(assign_cost[c]):
            if load[cand] + demands[c] <= cap[cand]:
                load[f] -= demands[c]
                load[cand] += demands[c]
                labels[c] = cand
                break
    return labels


def _local_search(labels, demands, capacities, open_cost, assign_cost, deadline, rng):
    n = len(demands)
    m = len(capacities)
    cap = np.asarray(capacities, dtype=float)
    load = np.zeros(m)
    np.add.at(load, labels, demands)
    used = np.zeros(m, dtype=bool)
    used[labels] = True
    cur = _total_cost(labels, demands, capacities, open_cost, assign_cost)

    def delta(c, nf):
        old = labels[c]
        if nf == old:
            return 0.0
        dc = assign_cost[c, nf] - assign_cost[c, old]
        if not used[old] and load[old] <= demands[c]:
            dc -= open_cost[old]
        if not used[nf] and load[nf] <= 0:
            dc += open_cost[nf]
        return dc

    def apply(c, nf):
        nonlocal cur
        old = labels[c]
        load[old] -= demands[c]
        if load[old] <= 0:
            used[old] = False
        load[nf] += demands[c]
        if not used[nf]:
            used[nf] = True
        labels[c] = nf
        cur += delta(c, nf)

    # Single-move passes.
    while time.time() < deadline:
        improved = False
        for c in range(n):
            if time.time() >= deadline:
                break
            old = labels[c]
            if load[old] <= demands[c]:
                feasible = [f for f in range(m) if f != old]
            else:
                feasible = [f for f in range(m)
                            if f != old and load[f] + demands[c] <= cap[f]]
            if not feasible:
                continue
            best_f = -1
            best_d = 1e18
            for f in feasible:
                dd = delta(c, f)
                if dd < best_d:
                    best_d = dd
                    best_f = f
            if best_f >= 0 and best_d < -1e-9:
                apply(c, best_f)
                improved = True
        if not improved:
            break

    # Pairwise swap passes.
    while time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                if time.time() >= deadline:
                    break
                fi = labels[i]
                fj = labels[j]
                if fi == fj:
                    continue
                if load[fi] - demands[i] < 0 or load[fj] - demands[j] < 0:
                    continue
                if load[fi] - demands[i] + demands[j] > cap[fi]:
                    continue
                if load[fj] - demands[j] + demands[i] > cap[fj]:
                    continue
                # compute delta of swapping i->fj and j->fi
                di = assign_cost[i, fj] - assign_cost[i, fi]
                dj = assign_cost[j, fi] - assign_cost[j, fj]
                dc = di + dj
                if not used[fi] and load[fi] <= demands[i]:
                    dc -= open_cost[fi]
                if not used[fj] and load[fj] <= demands[j]:
                    dc -= open_cost[fj]
                if not used[fi] and load[fi] - demands[i] + demands[j] > 0:
                    dc += open_cost[fi]
                if not used[fj] and load[fj] - demands[j] + demands[i] > 0:
                    dc += open_cost[fj]
                if dc < -1e-9:
                    load[fi] += demands[j] - demands[i]
                    load[fj] += demands[i] - demands[j]
                    if load[fi] <= 0:
                        used[fi] = False
                    if load[fj] <= 0:
                        used[fj] = False
                    labels[i] = fj
                    labels[j] = fi
                    cur += dc
                    improved = True
        if not improved:
            break
    return labels.copy(), cur


def assign(demands, capacities, open_cost, assign_cost):
    start = time.time()
    deadline = start + 140.0
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    n = len(demands)
    m = len(capacities)

    # Greedy fallback (always valid).
    greedy = np.zeros(n, dtype=int)
    gload = np.zeros(m)
    for i in np.argsort(-demands):
        for f in np.argsort(assign_cost[i]):
            if gload[f] + demands[i] <= capacities[f]:
                gload[f] += demands[i]
                greedy[i] = f
                break
    best = greedy
    best_cost = _total_cost(greedy, demands, capacities, open_cost, assign_cost)

    rng = np.random.default_rng(0)
    restarts = 8
    per = (deadline - start) / max(1, restarts)
    for r in range(restarts):
        t = start + r * per
        if t >= deadline - 5.0:
            break
        seed = int(rng.integers(0, 2**32))
        rr = np.random.default_rng(seed)
        labels = _construct(demands, capacities, open_cost, assign_cost, rr)
        if not _valid(labels, demands, capacities):
            continue
        labels, cost = _local_search(labels, demands, capacities, open_cost,
                                     assign_cost, t + per - 2.0, rr)
        if _valid(labels, demands, capacities):
            if cost < best_cost:
                best_cost = cost
                best = labels.copy()
    return best
# EVOLVE-BLOCK-END
