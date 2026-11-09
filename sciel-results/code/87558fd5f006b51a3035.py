"""Capacitated facility location: Multi-start ILS with vectorized moves + close-facility."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using multi-start local search with
    vectorized single moves, swap moves, and close-facility moves.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    rng = np.random.default_rng(42)

    deadline = time.time() + 142.0

    def calc_cost(res):
        used = np.unique(res)
        return float(np.sum(open_cost[used])) + float(np.sum(assign_cost[np.arange(n_cust), res]))

    def loads_of(res):
        load = np.zeros(n_fac)
        np.add.at(load, res, demands)
        return load

    # ---- Greedy seed (always valid fallback) ----
    result_greedy = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    order = np.argsort(-demands)
    for i in order:
        feasible = np.where(load + demands[i] <= capacities)[0]
        if len(feasible) == 0:
            f = int(np.argmax(capacities - load))
            result_greedy[i] = f
            load[f] += demands[i]
        else:
            costs = assign_cost[i, feasible]
            f_idx = int(feasible[np.argmin(costs)])
            result_greedy[i] = f_idx
            load[f_idx] += demands[i]

    best_result = result_greedy.copy()
    best_score = calc_cost(best_result)

    # ---- Vectorized single-move best-improvement pass ----
    def single_moves_pass(res, load):
        improved = False
        while True:
            if time.time() > deadline - 2.0:
                break
            any_improve = False
            base_assign = assign_cost[np.arange(n_cust), res]
            for i in range(n_cust):
                if time.time() > deadline - 2.0:
                    break
                f_old = res[i]
                d_i = demands[i]
                feasible_mask = (load + d_i <= capacities)
                feasible_mask = feasible_mask & (np.arange(n_fac) != f_old)
                if not np.any(feasible_mask):
                    continue
                delta_assign = assign_cost[i, :] - base_assign[i]
                delta_open = np.zeros(n_fac)
                if load[f_old] == d_i:
                    delta_open[f_old] = -open_cost[f_old]
                open_new = (load == 0) & (np.arange(n_fac) != f_old)
                if np.any(open_new):
                    delta_open[open_new] = open_cost[open_new]
                total_delta = delta_assign + delta_open
                total_delta[~feasible_mask] = np.inf
                total_delta[f_old] = np.inf
                best_f_new = int(np.argmin(total_delta))
                best_delta = total_delta[best_f_new]
                if best_delta < -1e-9:
                    res[i] = best_f_new
                    load[f_old] -= d_i
                    load[best_f_new] += d_i
                    any_improve = True
                    improved = True
            if not any_improve:
                break
        return res, load, improved

    # ---- Swap moves (sampled) ----
    def swap_pass(res, load, n_samples=2000):
        improved = False
        for _ in range(n_samples):
            if time.time() > deadline - 2.0:
                break
            i, j = rng.integers(0, n_cust, size=2)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            f_i = res[i]
            f_j = res[j]
            if f_i == f_j:
                continue
            d_i = demands[i]
            d_j = demands[j]
            if load[f_j] - d_j + d_i > capacities[f_j]:
                continue
            if load[f_i] - d_i + d_j > capacities[f_i]:
                continue
            delta_assign = (assign_cost[i, f_j] - assign_cost[i, f_i]) + \
                           (assign_cost[j, f_i] - assign_cost[j, f_j])
            total_delta = delta_assign
            if total_delta < -1e-9:
                res[i], res[j] = res[j], res[i]
                load[f_i] -= d_i
                load[f_i] += d_j
                load[f_j] -= d_j
                load[f_j] += d_i
                improved = True
        return res, load, improved

    # ---- Close-facility move ----
    def close_facility_pass(res, load):
        improved = False
        for f_close in range(n_fac):
            if time.time() > deadline - 2.0:
                break
            if load[f_close] == 0:
                continue
            custs = np.where(res == f_close)[0]
            if len(custs) == 0:
                continue

            temp_load = load.copy()
            temp_load[f_close] = 0
            temp_res = res.copy()

            cust_order = custs[np.argsort(-demands[custs])]
            success = True
            for c in cust_order:
                d_c = demands[c]
                feasible = np.where(temp_load + d_c <= capacities)[0]
                if len(feasible) == 0:
                    success = False
                    break
                costs = assign_cost[c, feasible]
                open_penalty = np.where(temp_load[feasible] == 0, open_cost[feasible], 0.0)
                adjusted_costs = costs + open_penalty
                best_f = feasible[np.argmin(adjusted_costs)]
                temp_res[c] = best_f
                temp_load[best_f] += d_c

            if not success:
                continue
            if np.any(temp_load > capacities):
                continue
            if temp_load[f_close] > 0:
                continue

            old_assign_sum = np.sum(assign_cost[custs, f_close])
            new_assign_sum = np.sum(assign_cost[custs, temp_res[custs]])

            newly_opened = set()
            for c in custs:
                nf = temp_res[c]
                if nf != f_close and temp_load[nf] > 0:
                    if load[nf] == 0:
                        newly_opened.add(nf)

            open_cost_change = -open_cost[f_close]
            for nf in newly_opened:
                open_cost_change += open_cost[nf]

            total_delta = (new_assign_sum - old_assign_sum) + open_cost_change

            if total_delta < -1e-9:
                for c in custs:
                    res[c] = temp_res[c]
                load = temp_load.copy()
                improved = True
        return res, load, improved

    # ---- Local search combining all moves ----
    def local_search(res, load):
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 2.0:
                break

            res, load, imp1 = single_moves_pass(res, load)
            improved = improved or imp1

            if time.time() > deadline - 2.0:
                break
            res, load, imp2 = swap_pass(res, load, n_samples=1500)
            improved = improved or imp2

            if time.time() > deadline - 2.0:
                break
            res, load, imp3 = close_facility_pass(res, load)
            improved = improved or imp3
        return res, load

    # ---- Multi-start with diverse initializations ----
    alphas = [0.0, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]

    for alpha in alphas:
        if time.time() > deadline - 10.0:
            break

        eff_cost = assign_cost + alpha * open_cost
        init_res = np.argmin(eff_cost, axis=1)

        res = init_res.copy()
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[res[i]] += demands[i]

        # Fix overloads
        max_iter = 300
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            overloaded = np.where(load > capacities)[0]
            if len(overloaded) == 0:
                break
            f = overloaded[0]
            custs_in_f = np.where(res == f)[0]
            if len(custs_in_f) == 0:
                break
            best_cust = -1
            best_g = -1
            min_penalty = np.inf
            for i in custs_in_f:
                for g in range(n_fac):
                    if g == f:
                        continue
                    if load[g] + demands[i] <= capacities[g]:
                        penalty = assign_cost[i, g] - assign_cost[i, f]
                        if penalty < min_penalty:
                            min_penalty = penalty
                            best_cust = i
                            best_g = g
            if best_cust == -1:
                break
            res[best_cust] = best_g
            load[f] -= demands[best_cust]
            load[best_g] += demands[best_cust]

        if np.all(load <= capacities):
            used = np.unique(res)
            score = np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(n_cust), res])
            if score < best_score:
                best_score = score
                best_result = res.copy()

                res_ls, load_ls = local_search(best_result.copy(), load.copy())
                if np.all(load_ls <= capacities):
                    used_ls = np.unique(res_ls)
                    score_ls = np.sum(open_cost[used_ls]) + np.sum(assign_cost[np.arange(n_cust), res_ls])
                    if score_ls < best_score:
                        best_score = score_ls
                        best_result = res_ls.copy()

    # Final local search on the best result found
    current_load = np.zeros(n_fac)
    for i in range(n_cust):
        current_load[best_result[i]] += demands[i]

    res_final, load_final = local_search(best_result.copy(), current_load.copy())
    if np.all(load_final <= capacities):
        used_final = np.unique(res_final)
        score_final = np.sum(open_cost[used_final]) + np.sum(assign_cost[np.arange(n_cust), res_final])
        if score_final < best_score:
            best_result = res_final
            best_score = score_final

    return best_result
# EVOLVE-BLOCK-END
