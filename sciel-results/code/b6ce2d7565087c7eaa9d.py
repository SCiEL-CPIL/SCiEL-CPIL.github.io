"""Capacitated facility location: Lagrangian-guided seed + local search with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def assign(demands, capacities, open_cost, assign_cost):
    t0 = time.time()
    DEADLINE = t0 + 145.0  # safety margin before 150s kill

    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    n_cust = len(demands)
    n_fac = len(capacities)

    # ---------- Objective helpers ----------
    def total_cost(result, load):
        open_mask = load > 0
        return float(np.sum(open_cost[open_mask]) +
                     np.sum(assign_cost[np.arange(n_cust), result]))

    def feasible(result):
        load = np.bincount(result, weights=demands, minlength=n_fac)
        return load <= capacities + 1e-9

    # ---------- Lagrangian-guided greedy ----------
    def lagrangian_seed(mu):
        # adjusted cost: assign_cost + mu * demand (penalty for consuming capacity)
        adj = assign_cost + np.outer(demands, mu)
        order = np.argsort(-demands)  # largest demand first
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        ok = True
        for i in order:
            # pick cheapest adjusted facility with capacity
            best_f = -1
            best_c = np.inf
            # vectorized argmin over feasible
            feasible_mask = load + demands[i] <= capacities + 1e-9
            if not np.any(feasible_mask):
                ok = False
                break
            costs = np.where(feasible_mask, adj[i], np.inf)
            f = int(np.argmin(costs))
            if costs[f] >= np.inf:
                ok = False
                break
            load[f] += demands[i]
            result[i] = f
        return result, load, ok

    # ---------- Subgradient update of mu ----------
    def subgradient_update(mu, load, step):
        # mu_f updated by step * (load_f - capacities_f)
        mu = mu + step * (load - capacities)
        return np.clip(mu, 0.0, None)

    # ---------- Local search ----------
    def local_search(result, load, max_iters=None):
        # Returns best (result, load, cost)
        if max_iters is None:
            max_iters = 200000
        cur_cost = total_cost(result, load)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > DEADLINE:
                break
            # Relocate moves: for each customer, best alternative facility
            # gain = (new_assign - old_assign) - (open_cost delta if closing a facility)
            # We evaluate relocate first (cheaper), then swaps.

            # For each customer i currently at f, compute best g != f that is feasible
            # and gives improvement including open-cost effect.
            # open_cost effect: moving i from f to g:
            #   if f becomes empty: -open_cost[f] (we close it)
            #   if g was empty: +open_cost[g] (we open it)
            # We approximate by computing per-customer best and checking.

            # Vectorized: for each i, the best facility g (g != result[i]) with capacity.
            # We'll do a batched check.
            best_gain = np.zeros(n_cust)
            best_target = result.copy()
            for i in range(n_cust):
                f = result[i]
                # capacity after removing i from f
                # trial load for g: load[g] + demands[i] <= capacities[g]
                # (f is being vacated by i, so f's load decreases by demands[i], but g's increases)
                feasible_g = (load + demands[i]) <= capacities + 1e-9
                feasible_g[f] = False
                if not np.any(feasible_g):
                    continue
                # cost delta for assign
                delta_assign = assign_cost[i, feasible_g] - assign_cost[i, f]
                # open cost delta:
                #   f becomes empty? load[f] - demands[i] <= 0  -> close f
                #   g becomes open? load[g] <= 0 (before adding) -> open g
                open_delta = np.zeros(n_fac)
                if load[f] - demands[i] <= 1e-9:
                    open_delta[f] = -open_cost[f]  # closing f
                # for each g, if load[g] <= 0 then opening g
                open_delta[load <= 1e-9] = open_cost[load[load <= 1e-9]]
                # but careful: g != f handled by mask
                total_delta = delta_assign + open_delta
                total_delta[~feasible_g] = np.inf
                g = int(np.argmin(total_delta))
                if total_delta[g] < -1e-9:
                    best_gain[i] = total_delta[g]
                    best_target[i] = g

            # Apply the best single improving relocate (or all non-conflicting? apply best)
            if np.any(best_gain < -1e-9):
                # pick the customer with the most negative gain
                i = int(np.argmin(best_gain))
                f = result[i]
                g = best_target[i]
                # verify feasibility
                if load[g] + demands[i] <= capacities[g] + 1e-9:
                    load[f] -= demands[i]
                    load[g] += demands[i]
                    result[i] = g
                    cur_cost += best_gain[i]
                    improved = True
                    continue

            # Swap moves: exchange two customers between facilities
            # For each pair (i,j) with result[i] != result[j], consider swapping.
            # gain = assign_cost[i, result[j]] + assign_cost[j, result[i]]
            #        - assign_cost[i, result[i]] - assign_cost[j, result[j]]
            #        + open_cost delta (facilities may open/close)
            # Feasibility: after swap, load at f_i = load[f_i] - d_i + d_j <= cap[f_i]
            #              load at f_j = load[f_j] - d_j + d_i <= cap[f_j]
            # We do a limited pass: sample pairs to keep it fast.
            # For efficiency, only consider swaps where both facilities have room.
            # We'll do a coarse pass over a subset of customers.

            # Build list of customers per facility for candidate selection
            # To keep it fast, sample up to a few hundred random pairs per iteration.
            if iters % 2 == 0:  # alternate relocate / swap
                n_sample = min(300, n_cust * n_cust // 2)
                for _ in range(n_sample):
                    if time.time() > DEADLINE:
                        break
                    i = int(np.random.randint(n_cust))
                    j = int(np.random.randint(n_cust))
                    if i == j:
                        continue
                    f = result[i]
                    g = result[j]
                    if f == g:
                        continue
                    # feasibility
                    if load[f] - demands[i] + demands[j] > capacities[f] + 1e-9:
                        continue
                    if load[g] - demands[j] + demands[i] > capacities[g] + 1e-9:
                        continue
                    # open cost delta
                    open_delta = 0.0
                    # f after: load[f] - d_i + d_j
                    # g after: load[g] - d_j + d_i
                    f_after_empty = (load[f] - demands[i] + demands[j]) <= 1e-9
                    g_after_empty = (load[g] - demands[j] + demands[i]) <= 1e-9
                    f_before_empty = load[f] <= 1e-9
                    g_before_empty = load[g] <= 1e-9
                    # open cost change:
                    # open set before: f open if load[f]>0, g open if load[g]>0
                    # open set after: f open if not f_after_empty, g open if not g_after_empty
                    before_open = (1.0 if load[f] > 1e-9 else 0.0) + (1.0 if load[g] > 1e-9 else 0.0)
                    after_open = (0.0 if f_after_empty else 1.0) + (0.0 if g_after_empty else 1.0)
                    open_delta = (after_open - before_open) * 0  # placeholder, compute properly below
                    # Proper: sum of open_cost over facilities whose open status changed
                    # Only f and g can change.
                    open_delta = 0.0
                    if load[f] > 1e-9 and f_after_empty:
                        open_delta += -open_cost[f]  # closing f
                    elif load[f] <= 1e-9 and not f_after_empty:
                        open_delta += open_cost[f]  # opening f
                    if load[g] > 1e-9 and g_after_empty:
                        open_delta += -open_cost[g]
                    elif load[g] <= 1e-9 and not g_after_empty:
                        open_delta += open_cost[g]

                    delta_assign = (assign_cost[i, g] + assign_cost[j, f]
                                    - assign_cost[i, f] - assign_cost[j, g])
                    total_delta = delta_assign + open_delta
                    if total_delta < -1e-9:
                        # apply swap
                        load[f] = load[f] - demands[i] + demands[j]
                        load[g] = load[g] - demands[j] + demands[i]
                        result[i] = g
                        result[j] = f
                        cur_cost += total_delta
                        improved = True
                        break  # restart relocate pass
        return result, load, cur_cost

    # ---------- Build initial solution via Lagrangian + subgradient ----------
    mu = np.zeros(n_fac)
    best_result = None
    best_load = None
    best_cost = np.inf

    # Initial greedy seed (mu=0)
    result0, load0, ok0 = lagrangian_seed(mu)
    if ok0 and feasible(result0):
        best_result = result0.copy()
        best_load = load0.copy()
        best_cost = total_cost(best_result, best_load)

    # Subgradient iterations
    step0 = 2.0
    for it in range(60):
        if time.time() > DEADLINE - 60.0:
            break
        result, load, ok = lagrangian_seed(mu)
        if ok and feasible(result):
            c = total_cost(result, load)
            if c < best_cost - 1e-9:
                best_cost = c
                best_result = result.copy()
                best_load = load.copy()
        # update mu
        if not ok:
            # infeasible: increase mu to penalize
            mu = subgradient_update(mu, load, step0)
        else:
            # subgradient of Lagrangian: (load - capacities)
            # use step that decreases over time
            step = step0 / (1.0 + 0.05 * it)
            mu = subgradient_update(mu, load, step)

    # Local search from best
    if best_result is not None and time.time() < DEADLINE - 10.0:
        best_result, best_load, best_cost = local_search(
            best_result.copy(), best_load.copy())

    # Final feasibility check; if infeasible, fall back to simple greedy
    if best_result is None or not feasible(best_result):
        # simple greedy fallback
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        for i in np.argsort(-demands):
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    load[f] += demands[i]
                    result[i] = f
                    break
        best_result = result

    return best_result.astype(int)
# EVOLVE-BLOCK-END
