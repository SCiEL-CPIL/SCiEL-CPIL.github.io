"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 138.0  # 12s safety margin before 150s kill

    n_customers = len(demands)
    n_facilities = len(capacities)

    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    def total_cost(res, load):
        used = np.zeros(n_facilities, dtype=bool)
        for f in np.unique(res):
            if f >= 0 and f < n_facilities:
                used[f] = True
        return float(open_cost[used].sum() + assign_cost[np.arange(n_customers), res].sum())

    def feasible(res, load):
        if np.any(res < 0) or np.any(res >= n_facilities):
            return False
        return bool(np.all(load <= capacities))

    def make_feasible(res, load):
        """Repair capacity violations by moving customers to under-capacity facilities."""
        res = res.copy()
        load = load.copy()
        order = np.argsort(-demands)
        for _ in range(n_customers * 4):
            over = np.where(load > capacities)[0]
            if len(over) == 0:
                break
            worst = over[np.argmax(load[over] - capacities[over])]
            custs = np.where(res == worst)[0]
            moved = False
            for i in custs:
                rem = capacities - load
                rem[worst] = 0
                rem[res[i]] = 0
                cands = np.where(rem >= demands[i])[0]
                if len(cands) > 0:
                    f = cands[np.argmin(assign_cost[i, cands])]
                    load[worst] -= demands[i]
                    load[f] += demands[i]
                    res[i] = f
                    moved = True
                    break
            if not moved:
                break
        return res, load

    def local_search_1move(res, load, t0):
        """Repeatedly apply all improving 1-moves (vectorized)."""
        while True:
            if time.time() > t0:
                break
            cur = assign_cost[np.arange(n_customers), res]
            best_other = assign_cost.copy()
            best_other[np.arange(n_customers), res] = np.inf
            # capacity mask: facility f feasible for customer i if load[f]+d[i] <= cap[f]
            feasible = (load[None, :] + demands[:, None]) <= capacities[None, :]
            best_other[~feasible] = np.inf
            # exclude current facility (already inf)
            best = np.argmin(best_other, axis=1)
            best_val = np.min(best_other, axis=1)
            new_cost = best_val - cur
            # open cost: opening a previously-empty target
            target_empty = (load[best] == 0)
            new_cost += np.where(target_empty, open_cost[best], 0.0)
            # open cost: source becomes empty
            src = res
            src_empty_after = np.zeros(n_customers, dtype=bool)
            for f in np.unique(src):
                members = np.where(src == f)[0]
                if len(members) == 1:
                    src_empty_after[members[0]] = True
            new_cost += np.where(src_empty_after, -open_cost[src], 0.0)

            improving = np.where(new_cost < -1e-9)[0]
            if len(improving) == 0:
                break
            # apply all improving moves (safe: distinct customers, capacity checked)
            for i in improving:
                f = best[i]
                load[src[i]] -= demands[i]
                load[f] += demands[i]
                res[i] = f
        return res, load

    def facility_elimination(res, load, t0):
        """Try to close each facility by moving all its customers elsewhere."""
        improved = True
        while improved:
            if time.time() > t0:
                break
            improved = False
            for f in range(n_facilities):
                if time.time() > t0:
                    break
                members = np.where(res == f)[0]
                if len(members) == 0:
                    continue
                # total assignment cost to move all members off f
                # find a feasible target for each member (not f)
                rem_after = capacities - load
                rem_after[f] = 0
                ok = True
                new_res = res.copy()
                new_load = load.copy()
                # move members one by one, greedy cheapest feasible
                # process larger demands first within this group
                order = members[np.argsort(-demands[members])]
                for i in order:
                    rem = new_load and (capacities - new_load)
                    rem = capacities - new_load
                    rem[f] = 0
                    cands = np.where(rem >= demands[i])[0]
                    if len(cands) == 0:
                        ok = False
                        break
                    tf = cands[np.argmin(assign_cost[i, cands])]
                    new_load[f] -= demands[i]
                    new_load[tf] += demands[i]
                    new_res[i] = tf
                if not ok:
                    continue
                # cost comparison
                old_open = open_cost[f] if len(members) > 0 else 0.0
                new_res_set = set(np.unique(new_res).tolist())
                old_res_set = set(np.unique(res).tolist())
                # open cost delta
                opened = new_res_set - old_res_set
                closed = old_res_set - new_res_set
                open_delta = sum(open_cost[x] for x in opened) - sum(open_cost[x] for x in closed)
                assign_delta = float(assign_cost[np.arange(n_customers), new_res].sum()
                                     - assign_cost[np.arange(n_customers), res].sum())
                if open_delta + assign_delta < -1e-9:
                    res = new_res
                    load = new_load
                    improved = True
        return res, load

    best_res = None
    best_load = None
    best_cost = np.inf

    # Restart-based vectorized initialization
    n_restarts = 24
    for r in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        rng = np.random.default_rng(1000 + r)
        res = np.full(n_customers, -1, dtype=np.int64)
        load = np.zeros(n_facilities, dtype=np.int64)
        order = np.argsort(-demands)
        for i in order:
            rem = capacities - load
            cands = np.where(rem >= demands[i])[0]
            if len(cands) == 0:
                # pick facility with most remaining capacity (will repair later)
                f = int(np.argmax(rem))
            else:
                # random among feasible, weighted toward cheaper
                probs = 1.0 / (assign_cost[i, cands] + 1.0)
                probs /= probs.sum()
                f = int(cands[rng.choice(len(cands), p=probs)])
            res[i] = f
            load[f] += demands[i]
        res, load = make_feasible(res, load)
        if not feasible(res, load):
            continue
        res, load = local_search_1move(res, load, deadline - 25.0)
        res, load = facility_elimination(res, load, deadline - 20.0)
        # a final polish 1-move pass
        res, load = local_search_1move(res, load, deadline - 10.0)
        c = total_cost(res, load)
        if c < best_cost - 1e-9:
            best_cost = c
            best_res = res.copy()
            best_load = load.copy()

    if best_res is None:
        # Fallback: pure greedy
        res = np.zeros(n_customers, dtype=np.int64)
        load = np.zeros(n_facilities, dtype=np.int64)
        for i in np.argsort(-demands):
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    res[i] = f
                    break
        best_res = res
        best_load = load

    # Final feasibility guard
    fl = np.zeros(n_facilities, dtype=np.int64)
    for i in range(n_customers):
        fl[best_res[i]] += demands[i]
    if np.any(fl > capacities):
        best_res, best_load = make_feasible(best_res, best_load)

    return best_res.astype(np.int64)
# EVOLVE-BLOCK-END
