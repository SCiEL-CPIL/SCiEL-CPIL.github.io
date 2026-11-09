# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    n_cust = len(demands)
    n_fac = len(capacities)

    start_time = time.time()
    deadline = start_time + 140.0

    def calc_cost(res):
        used = np.unique(res)
        if len(used) == 0:
            return 0.0
        open_sum = float(np.sum(open_cost[used]))
        assign_sum = float(np.sum(assign_cost[np.arange(n_cust), res]))
        return open_sum + assign_sum

    def loads_of(res):
        load = np.zeros(n_fac, dtype=np.int64)
        np.add.at(load, res, demands)
        return load

    def _greedy(order, allowed):
        allowed_arr = np.asarray(allowed, dtype=np.int64)
        n_a = len(allowed_arr)
        if n_a == 0:
            return None
        if int(np.sum(capacities[allowed_arr])) < int(np.sum(demands)):
            return None
        load = np.zeros(n_a, dtype=np.int64)
        res = np.full(n_cust, -1, dtype=np.int64)
        sub_cost = assign_cost[:, allowed_arr]
        sorted_f = np.argsort(sub_cost, axis=1)
        for i in order:
            placed = False
            for fl in sorted_f[i]:
                if load[fl] + demands[i] <= capacities[allowed_arr[fl]]:
                    load[fl] += demands[i]
                    res[i] = allowed_arr[fl]
                    placed = True
                    break
            if not placed:
                return None
        return res

    def _local_search_vectorized(res, load, max_time):
        """Vectorized local search with single moves, swaps, and facility closing."""
        res = res.copy()
        load = load.copy()
        t0 = time.time()
        rng = np.random.default_rng(42)

        improved = True
        while improved and (time.time() - t0) < max_time and time.time() < deadline - 3.0:
            improved = False

            # --- Vectorized single moves ---
            # For each customer, compute the best feasible move
            d_all = demands
            # Precompute feasibility mask for each customer to each facility
            # mask[i, f] = (load[f] + d_all[i] <= capacities[f]) and f != res[i]
            feasible = (load[None, :] + d_all[:, None] <= capacities[None, :]).astype(np.int8)
            # Exclude current facility
            for i in range(n_cust):
                feasible[i, res[i]] = 0

            # Cost delta for moving customer i from res[i] to f
            # delta = assign_cost[i, f] - assign_cost[i, res[i]]
            # Plus open cost adjustments:
            #   If load[res[i]] == d_all[i], f_old becomes empty: save open_cost[res[i]]
            #   If load[f] == 0, f newly opened: pay open_cost[f]
            base_costs = assign_cost[np.arange(n_cust), res]
            cost_diff = assign_cost - base_costs[:, None]

            # Open cost adjustments
            # Save open cost if old facility becomes empty
            save_open = np.zeros((n_cust, n_fac), dtype=np.float64)
            for i in range(n_cust):
                if load[res[i]] == d_all[i]:
                    save_open[i, :] = -open_cost[res[i]]

            # Pay open cost if new facility was empty
            pay_open = np.zeros((n_cust, n_fac), dtype=np.float64)
            empty_fac = (load == 0)
            for i in range(n_cust):
                pay_open[i, empty_fac] = open_cost[empty_fac]

            total_delta = cost_diff + save_open + pay_open
            total_delta[~feasible.astype(bool)] = np.inf

            # Find best move for each customer
            best_delta = np.min(total_delta, axis=1)
            best_fac = np.argmin(total_delta, axis=1)

            # Apply all improving moves (with care for conflicts)
            # Process in order of most negative delta
            order = np.argsort(best_delta)
            applied = 0
            for i in order:
                if time.time() > deadline - 3.0:
                    break
                if best_delta[i] >= -1e-9:
                    break
                f_old = res[i]
                f_new = best_fac[i]
                d_i = d_all[i]
                if load[f_new] + d_i > capacities[f_new]:
                    continue
                res[i] = f_new
                load[f_old] -= d_i
                load[f_new] += d_i
                improved = True
                applied += 1

            if not improved:
                # --- Sampled swap moves ---
                n_swaps = 2000
                for _ in range(n_swaps):
                    if time.time() > deadline - 3.0:
                        break
                    i, j = int(rng.integers(0, n_cust)), int(rng.integers(0, n_cust))
                    if i == j:
                        continue
                    f_i, f_j = res[i], res[j]
                    if f_i == f_j:
                        continue
                    d_i, d_j = d_all[i], d_all[j]
                    new_load_i = load[f_i] - d_i + d_j
                    new_load_j = load[f_j] - d_j + d_i
                    if new_load_i > capacities[f_i] or new_load_j > capacities[f_j]:
                        continue
                    delta_assign = (assign_cost[i, f_j] - assign_cost[i, f_i]) + \
                                   (assign_cost[j, f_i] - assign_cost[j, f_j])
                    old_open_i = open_cost[f_i] if load[f_i] > 0 else 0.0
                    old_open_j = open_cost[f_j] if load[f_j] > 0 else 0.0
                    new_open_i = open_cost[f_i] if new_load_i > 0 else 0.0
                    new_open_j = open_cost[f_j] if new_load_j > 0 else 0.0
                    delta_open = (new_open_i + new_open_j) - (old_open_i + old_open_j)
                    if delta_assign + delta_open < -1e-9:
                        res[i], res[j] = res[j], res[i]
                        load[f_i] = new_load_i
                        load[f_j] = new_load_j
                        improved = True

                # --- Facility-closing moves ---
                for f_close in range(n_fac):
                    if time.time() > deadline - 3.0:
                        break
                    if load[f_close] == 0:
                        continue
                    custs = np.where(res == f_close)[0]
                    if len(custs) == 0:
                        continue
                    # Try to reassign all customers from f_close to other facilities
                    temp_load = load.copy()
                    temp_res = res.copy()
                    temp_load[f_close] = 0
                    success = True
                    # Sort by demand descending for better packing
                    for c in np.argsort(-d_all[custs]):
                        cc = custs[c]
                        placed = False
                        for f in np.argsort(assign_cost[cc]):
                            if f == f_close:
                                continue
                            if temp_load[f] + d_all[cc] <= capacities[f]:
                                temp_load[f] += d_all[cc]
                                temp_res[cc] = f
                                placed = True
                                break
                        if not placed:
                            success = False
                            break
                    if success:
                        new_cost = calc_cost(temp_res)
                        old_cost = calc_cost(res)
                        if new_cost < old_cost - 1e-9:
                            res = temp_res
                            load = temp_load
                            improved = True

        return res, load

    # ---------- candidate facility subsets ----------
    avg_assign = assign_cost.mean(axis=0)
    by_open = np.argsort(open_cost)
    by_assign = np.argsort(avg_assign)
    by_combined = np.argsort(open_cost + avg_assign)

    candidate_subsets = []
    candidate_subsets.append(list(range(n_fac)))
    for k in (5, 8, 10, 12, 15, 18):
        if k <= n_fac:
            candidate_subsets.append(by_open[:k].tolist())
            candidate_subsets.append(by_assign[:k].tolist())
            candidate_subsets.append(by_combined[:k].tolist())

    # De-duplicate
    seen = set()
    unique_subsets = []
    for s in candidate_subsets:
        key = frozenset(s)
        if key not in seen:
            seen.add(key)
            unique_subsets.append(s)

    # ---------- multi-start greedy initialisations ----------
    orderings = [
        np.argsort(-demands),
        np.argsort(assign_cost.mean(axis=1)),
        np.arange(n_cust),
        np.random.default_rng(7).permutation(n_cust),
        np.random.default_rng(13).permutation(n_cust),
        np.random.default_rng(99).permutation(n_cust),
        np.argsort(demands * assign_cost.mean(axis=1)),
    ]

    best_result = None
    best_cost = float('inf')

    # Phase A: diverse greedy starts with local search
    for subset in unique_subsets:
        if time.time() > deadline - 30.0:
            break
        for order in orderings:
            if time.time() > deadline - 10.0:
                break
            res = _greedy(order, subset)
            if res is None:
                continue
            load = loads_of(res)
            # Short local search
            res, load = _local_search_vectorized(res, load, max_time=1.5)
            c = calc_cost(res)
            if c < best_cost:
                best_cost = c
                best_result = res.copy()

    # Phase B: deeper local search from best solution
    if best_result is not None and time.time() < deadline - 5.0:
        load = loads_of(best_result)
        remaining = deadline - 5.0 - time.time()
        res, load = _local_search_vectorized(best_result, load, max_time=remaining)
        c = calc_cost(res)
        if c < best_cost:
            best_cost = c
            best_result = res.copy()

    # ---------- guaranteed-valid fallback ----------
    if best_result is None:
        best_result = np.zeros(n_cust, dtype=np.int64)
        load = np.zeros(n_fac, dtype=np.int64)
        for i in np.argsort(-demands):
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    best_result[i] = f
                    break
        for i in range(n_cust):
            if best_result[i] < 0 or best_result[i] >= n_fac:
                for f in range(n_fac):
                    if load[f] + demands[i] <= capacities[f]:
                        load[f] += demands[i]
                        best_result[i] = f
                        break

    return best_result
# EVOLVE-BLOCK-END
