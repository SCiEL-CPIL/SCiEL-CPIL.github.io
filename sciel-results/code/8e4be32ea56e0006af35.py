"""Capacitated facility location: greedy init + large neighborhood search (LNS)."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def _total_cost(result, demands, capacities, open_cost, assign_cost):
    total = float(assign_cost[np.arange(len(demands)), result].sum())
    load = np.bincount(result, weights=demands, minlength=len(capacities))
    for f in range(len(capacities)):
        if load[f] > 0:
            total += open_cost[f]
    return total


def _greedy_init(demands, capacities, open_cost, assign_cost, rng):
    n = len(demands)
    m = len(capacities)
    result = np.zeros(n, dtype=int)
    load = np.zeros(m)
    order = np.argsort(-demands)
    for i in order:
        d = demands[i]
        # candidate facilities that can fit this demand
        feasible = np.where(load + d <= capacities)[0]
        if len(feasible) == 0:
            # fallback: any facility (should not happen if total capacity >= total demand)
            feasible = np.arange(m)
        # effective cost = assign_cost + expected open-cost share
        eff = assign_cost[i, feasible].copy()
        open_share = np.where(load[feasible] == 0, open_cost[feasible] / max(d, 1), 0.0)
        eff += open_share
        # with small probability pick a random feasible facility to diversify
        if rng.random() < 0.15:
            f = feasible[rng.randint(len(feasible))]
        else:
            f = feasible[np.argmin(eff)]
        load[f] += d
        result[i] = f
    return result


def _try_swap(i, j, result, demands, capacities, open_cost, assign_cost, load):
    """Return cost delta if moving customer i to facility of j (swap) is feasible, else None."""
    fi = result[i]
    fj = result[j]
    if fi == fj:
        return None
    di = demands[i]
    dj = demands[j]
    # after swap: fi loses di, fj loses dj; fi gains dj, fj gains di
    if load[fi] - di + dj > capacities[fi]:
        return None
    if load[fj] - dj + di > capacities[fj]:
        return None
    delta = (assign_cost[i, fj] + assign_cost[j, fi]) - (assign_cost[i, fi] + assign_cost[j, fj])
    # open cost change
    if load[fi] == di:  # fi becomes empty
        delta -= open_cost[fi]
    if load[fj] == dj:  # fj becomes empty
        delta -= open_cost[fj]
    if load[fi] - di == 0 and dj > 0:  # fi was empty, now has dj -> but fi had di>0 so not empty
        pass
    # more careful open cost:
    # fi had load[fi], after swap load[fi]-di+dj
    # fj had load[fj], after swap load[fj]-dj+di
    return delta


def _improve_swap(result, demands, capacities, open_cost, assign_cost, deadline):
    """Local search with swap moves. Returns improved result and cost."""
    n = len(demands)
    m = len(capacities)
    load = np.bincount(result, weights=demands, minlength=m).astype(float)
    cur_cost = _total_cost(result, demands, capacities, open_cost, assign_cost)
    improved = True
    while improved:
        improved = False
        # random order of customers for diversity
        order = np.random.permutation(n)
        for i in order:
            if time.time() > deadline:
                return result, cur_cost
            fi = result[i]
            di = demands[i]
            # find best swap partner
            # only consider facilities where moving i is feasible
            # try swapping with a customer in another facility
            # For speed, sample a few candidates
            # Find facilities that can accept di
            # load[fi] - di + dj <= cap[fi] -> dj <= cap[fi] - load[fi] + di
            # load[fj] - dj + di <= cap[fj]
            # We'll sample random customers from other facilities
            for _ in range(8):
                j = np.random.randint(n)
                if j == i:
                    continue
                fj = result[j]
                if fj == fi:
                    continue
                dj = demands[j]
                if load[fi] - di + dj > capacities[fi]:
                    continue
                if load[fj] - dj + di > capacities[fj]:
                    continue
                delta = (assign_cost[i, fj] + assign_cost[j, fi]) - (assign_cost[i, fi] + assign_cost[j, fj])
                # open cost
                new_load_fi = load[fi] - di + dj
                new_load_fj = load[fj] - dj + di
                old_fi_empty = (load[fi] == 0)
                old_fj_empty = (load[fj] == 0)
                new_fi_empty = (new_load_fi == 0)
                new_fj_empty = (new_load_fj == 0)
                if old_fi_empty and not new_fi_empty:
                    delta += open_cost[fi]
                if not old_fi_empty and new_fi_empty:
                    delta -= open_cost[fi]
                if old_fj_empty and not new_fj_empty:
                    delta += open_cost[fj]
                if not old_fj_empty and new_fj_empty:
                    delta -= open_cost[fj]
                if delta < -1e-9:
                    # accept swap
                    result[i] = fj
                    result[j] = fi
                    load[fi] = new_load_fi
                    load[fj] = new_load_fj
                    cur_cost += delta
                    improved = True
                    break
    return result, cur_cost


def _lns_destroy_repair(result, demands, capacities, open_cost, assign_cost, load, rng, deadline):
    """Remove a random subset of customers, reassign them greedily, keep if better."""
    n = len(demands)
    m = len(capacities)
    cur_cost = _total_cost(result, demands, capacities, open_cost, assign_cost)

    # destroy: remove 15-25% of customers
    k = max(1, int(n * rng.uniform(0.15, 0.25)))
    removed = rng.choice(n, size=k, replace=False)
    removed_set = set(removed.tolist())

    # free capacity
    for i in removed:
        load[result[i]] -= demands[i]

    # save old assignments
    old_assign = {i: result[i] for i in removed}

    # repair: assign removed customers one by one (largest first)
    removed_sorted = sorted(removed, key=lambda i: -demands[i])
    new_cost = cur_cost
    # recompute cost without removed customers
    base_cost = 0.0
    for i in range(n):
        if i not in removed_set:
            base_cost += assign_cost[i, result[i]]
    base_load = load.copy()
    for f in range(m):
        if base_load[f] > 0:
            base_cost += open_cost[f]

    for i in removed_sorted:
        d = demands[i]
        feasible = np.where(load + d <= capacities)[0]
        if len(feasible) == 0:
            feasible = np.arange(m)
        eff = assign_cost[i, feasible].copy()
        open_share = np.where(load[feasible] == 0, open_cost[feasible] / max(d, 1), 0.0)
        eff += open_share
        f = feasible[np.argmin(eff)]
        result[i] = f
        load[f] += d
        new_cost += assign_cost[i, f]
        if load[f] == d and open_cost[f] > 0:
            # check if this facility was empty before
            pass

    # recompute full cost properly
    new_cost = _total_cost(result, demands, capacities, open_cost, assign_cost)
    if new_cost < cur_cost - 1e-9:
        return result, new_cost, load
    else:
        # revert
        for i in removed:
            result[i] = old_assign[i]
        load = np.bincount(result, weights=demands, minlength=m).astype(float)
        return result, cur_cost, load


def assign(demands, capacities, open_cost, assign_cost):
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 120s limit
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    n = len(demands)
    m = len(capacities)
    rng = np.random.default_rng(42)

    # Multi-start: try several greedy inits, keep best
    best_result = None
    best_cost = np.inf

    n_starts = 8
    for s in range(n_starts):
        if time.time() > deadline:
            break
        res = _greedy_init(demands, capacities, open_cost, assign_cost, rng)
        # quick swap improvement
        res, cost = _improve_swap(res, demands, capacities, open_cost, assign_cost, deadline)
        if cost < best_cost:
            best_cost = cost
            best_result = res.copy()

    # LNS phase on the best solution
    if best_result is not None and time.time() < deadline:
        load = np.bincount(best_result, weights=demands, minlength=m).astype(float)
        cur_cost = best_cost
        while time.time() < deadline:
            # LNS destroy-repair
            best_result, cur_cost, load = _lns_destroy_repair(
                best_result, demands, capacities, open_cost, assign_cost, load, rng, deadline)
            # quick swap polish
            if time.time() < deadline:
                best_result, cur_cost = _improve_swap(
                    best_result, demands, capacities, open_cost, assign_cost, deadline)
                load = np.bincount(best_result, weights=demands, minlength=m).astype(float)

    if best_result is None:
        # ultimate fallback
        best_result = np.zeros(n, dtype=int)
        load = np.zeros(m)
        for i in np.argsort(-demands):
            for f in range(m):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    best_result[i] = f
                    break

    return best_result.astype(int)
# EVOLVE-BLOCK-END
