"""Capacitated facility location: Lagrangian warm-start + local search polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _deadline_guard():
    return time.time() < _DEADLINE

_DEADLINE = None

def assign(demands, capacities, open_cost, assign_cost):
    global _DEADLINE
    _DEADLINE = time.time() + 140.0  # safety margin vs 150s wall-clock

    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    n_cust = len(demands)
    n_fac = len(capacities)

    # ---------- Lagrangian relaxation warm start ----------
    # Objective: sum_f open_cost[f]*y_f + sum_i,j c_ij x_ij
    # s.t. sum_j x_ij = 1, sum_i d_i x_ij <= cap_j y_j, x,y binary
    # Relax capacity: penalty lambda_j for violation.
    # Subproblem per customer: min_j (c_ij + max(0, lambda_j)) if we treat
    # the capacity penalty additively per assignment (approx).

    def lagrangian(lambda_):
        # effective cost for customer i -> facility j
        eff = assign_cost + lambda_[None, :]
        best_j = np.argmin(eff, axis=1)
        # load per facility
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[best_j[i]] += demands[i]
        # objective (relaxed): assignment cost + open cost for used facs + penalty
        used = load > 0
        obj = np.sum(assign_cost[np.arange(n_cust), best_j]) \
              + np.sum(open_cost[used]) \
              + np.sum(np.maximum(0, load - capacities) * lambda_)
        return obj, best_j, load

    # Subgradient ascent on lambda
    lambda_ = np.zeros(n_fac)
    best_lambda_obj = np.inf
    best_lambda_j = None
    best_lambda_load = None

    # step size via Polyak-ish
    best_obj_any = np.inf
    for it in range(400):
        if not _deadline_guard():
            break
        obj, j, load = lagrangian(lambda_)
        if obj < best_obj_any:
            best_obj_any = obj
            best_lambda_j = j.copy()
            best_lambda_load = load.copy()
        # subgradient: violation
        grad = np.maximum(0, load - capacities)
        # simple step decay
        step = 2.0 / (it + 1.0)
        # normalize
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-9:
            break
        lambda_ = np.maximum(0, lambda_ + step * grad / gnorm * 200.0)

    # ---------- Build feasible initial solution from warm start ----------
    # Greedy repair: sort by demand desc, assign to cheapest facility with room,
    # preferring facilities suggested by Lagrangian.
    def build_feasible(order_fac_pref):
        # order_fac_pref: for each customer, ordered list of facility indices
        result = np.full(n_cust, -1, dtype=int)
        load = np.zeros(n_fac)
        for i in np.argsort(-demands):
            placed = False
            for f in order_fac_pref[i]:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    placed = True
                    break
            if not placed:
                # fallback: any facility with room
                for f in range(n_fac):
                    if load[f] + demands[i] <= capacities[f]:
                        load[f] += demands[i]
                        result[i] = f
                        placed = True
                        break
            if not placed:
                # infeasible; put in least-loaded
                f = int(np.argmin(load))
                load[f] += demands[i]
                result[i] = f
        return result, load

    # preference order: blend Lagrangian best facility + cost rank
    lag_pref = np.argsort(assign_cost + best_lambda_load[None, :] * 0.0, axis=1)
    # Actually use the lagrangian chosen facility as top, then cost order
    pref = np.argsort(assign_cost, axis=1)
    # put lagrangian choice first
    pref2 = pref.copy()
    for i in range(n_cust):
        lj = best_lambda_j[i] if best_lambda_j is not None else pref[i][0]
        # move lj to front
        idx = np.where(pref2[i] == lj)[0]
        if len(idx) and idx[0] > 0:
            pref2[i] = np.concatenate(([lj], np.delete(pref2[i], idx[0])))

    result, load = build_feasible(pref2)

    # ---------- Local search polish ----------
    # 1-opt: move a customer to a better facility if capacity allows
    # 2-opt: swap two customers between facilities

    def total_cost(res):
        used = np.unique(res[res >= 0])
        return np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(n_cust), res])

    cur_cost = total_cost(result)
    best_result = result.copy()
    best_cost = cur_cost

    # precompute
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]

    improved = True
    pass_count = 0
    while improved and _deadline_guard():
        improved = False
        pass_count += 1
        # 1-opt moves
        order = np.argsort(assign_cost[np.arange(n_cust), result] - assign_cost.mean(axis=1))
        for i in range(n_cust):
            if not _deadline_guard():
                break
            cur_f = result[i]
            # try better facilities
            cands = np.argsort(assign_cost[i])
            for f in cands:
                if f == cur_f:
                    continue
                if assign_cost[i, f] >= assign_cost[i, cur_f]:
                    break
                if load[f] + demands[i] <= capacities[f]:
                    # evaluate delta: assignment change + possible open cost change
                    old_open = open_cost[cur_f] if (load[cur_f] - demands[i]) <= 0 else 0.0
                    new_open = open_cost[f] if load[f] == 0 else 0.0
                    delta = (assign_cost[i, f] - assign_cost[i, cur_f]) + new_open - old_open
                    if delta < -1e-9:
                        load[cur_f] -= demands[i]
                        load[f] += demands[i]
                        result[i] = f
                        cur_cost += delta
                        improved = True
                        break
        # 2-opt swaps (limited)
        if _deadline_guard() and pass_count < 40:
            # group customers by facility
            for fa in range(n_fac):
                if not _deadline_guard():
                    break
                custs_a = np.where(result == fa)[0]
                if len(custs_a) < 2:
                    continue
                for fb in range(n_fac):
                    if fb == fa:
                        continue
                    custs_b = np.where(result == fb)[0]
                    if len(custs_b) < 2:
                        continue
                    # try swap one from a to b and one from b to a
                    for i in custs_a[:3]:
                        if not _deadline_guard():
                            break
                        for j in custs_b[:3]:
                            # swap i<->j
                            new_load_a = load[fa] - demands[i] + demands[j]
                            new_load_b = load[fb] - demands[j] + demands[i]
                            if new_load_a > capacities[fa] or new_load_b > capacities[fb]:
                                continue
                            old_open_a = open_cost[fa] if new_load_a <= 0 else 0.0
                            old_open_b = open_cost[fb] if new_load_b <= 0 else 0.0
                            # current open contribution
                            cur_open_a = open_cost[fa] if load[fa] <= 0 else 0.0
                            cur_open_b = open_cost[fb] if load[fb] <= 0 else 0.0
                            delta = (assign_cost[i, fb] - assign_cost[i, fa]) \
                                    + (assign_cost[j, fa] - assign_cost[j, fb]) \
                                    + (old_open_a + old_open_b) - (cur_open_a + cur_open_b)
                            if delta < -1e-9:
                                load[fa] = new_load_a
                                load[fb] = new_load_b
                                result[i] = fb
                                result[j] = fa
                                cur_cost += delta
                                improved = True
                                # recompute custs after change is complex; just break inner
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

        if cur_cost < best_cost - 1e-9:
            best_cost = cur_cost
            best_result = result.copy()

    # Final feasibility check & repair
    def is_valid(res):
        if np.any(res < 0) or np.any(res >= n_fac):
            return False
        ld = np.zeros(n_fac)
        for i in range(n_cust):
            ld[res[i]] += demands[i]
        return np.all(ld <= capacities + 1e-6)

    if not is_valid(best_result):
        # rebuild feasible
        result, load = build_feasible(pref2)
        best_result = result
        best_cost = total_cost(result)

    return best_result.astype(int)
# EVOLVE-BLOCK-END
