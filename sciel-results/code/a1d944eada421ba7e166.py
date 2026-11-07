import numpy as np
import time


def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def _time_ok():
        return time.time() < deadline

    n_customers = len(demands)
    n_facilities = len(capacities)

    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    INF = float('inf')

    def total_cost(result):
        load = np.zeros(n_facilities)
        for i, f in enumerate(result):
            load[f] += demands[i]
        if np.any(load > capacities + 1e-9):
            return INF
        open_mask = np.zeros(n_facilities, dtype=bool)
        for f in result:
            open_mask[f] = True
        open_sum = float(np.sum(open_cost[open_mask]))
        assign_sum = float(np.sum(assign_cost[np.arange(n_customers), result]))
        return open_sum + assign_sum

    def greedy_init(seed=None, penalty=None):
        """Greedy: process customers by demand desc, pick min marginal cost.
        penalty: optional per-facility extra cost used to diversify restarts."""
        if seed is not None:
            np.random.seed(seed)

        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        is_open = np.zeros(n_facilities, dtype=bool)

        order = np.argsort(-demands)
        pen = np.zeros(n_facilities) if penalty is None else np.asarray(penalty, dtype=np.float64)

        for i in order:
            d = demands[i]
            feasible = load + d <= capacities

            # marginal cost = assign_cost + open_cost if closing + penalty
            marg = assign_cost[i].copy()
            marg[~is_open] += open_cost[~is_open]
            marg += pen
            marg[~feasible] = INF

            if not np.any(np.isfinite(marg)):
                # relax: pick min assign cost among any facility
                f = int(np.argmin(assign_cost[i]))
            else:
                if seed is not None and np.random.rand() < 0.3:
                    # random among top few finite
                    finite = np.where(np.isfinite(marg))[0]
                    vals = marg[finite]
                    k = min(3, len(finite))
                    idx = np.argsort(vals)[:k]
                    f = int(finite[idx[np.random.randint(len(idx))]])
                else:
                    f = int(np.argmin(marg))

            result[i] = f
            load[f] += d
            is_open[f] = True

        return result, load, is_open

    def relocation_search(result, load, is_open, max_passes=20):
        """1-move: relocate a single customer to a better facility."""
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for i in range(n_customers):
                f_old = result[i]
                d = demands[i]

                # marginal cost of moving i to each f
                delta_assign = assign_cost[i] - assign_cost[i, f_old]
                # open cost change
                delta_open = np.zeros(n_facilities)
                # if old becomes empty, save its open cost (applies to all f)
                if load[f_old] - d <= 1e-9:
                    delta_open[:] = -open_cost[f_old]
                # if new was empty, add its open cost
                empty_new = (load == 0)
                delta_open[empty_new] += open_cost[empty_new]

                total = delta_assign + delta_open
                # capacity feasibility
                total[load + d > capacities] = INF
                total[f_old] = INF

                f_new = int(np.argmin(total))
                if total[f_new] < -1e-9:
                    # apply
                    load[f_old] -= d
                    if load[f_old] <= 1e-9:
                        load[f_old] = 0.0
                        is_open[f_old] = False
                    load[f_new] += d
                    if not is_open[f_new]:
                        is_open[f_new] = True
                    result[i] = f_new
                    improved = True
                    if not _time_ok():
                        return result, load, is_open
        return result, load, is_open

    def swap_search(result, load, is_open, max_passes=5):
        """2-move: swap two customers between facilities."""
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for i in range(n_customers):
                if not _time_ok():
                    return result, load, is_open
                f_i = result[i]
                d_i = demands[i]
                for j in range(i + 1, n_customers):
                    f_j = result[j]
                    if f_i == f_j:
                        continue
                    d_j = demands[j]
                    # capacity after swap
                    if load[f_i] - d_i + d_j > capacities[f_i] + 1e-9:
                        continue
                    if load[f_j] - d_j + d_i > capacities[f_j] + 1e-9:
                        continue
                    delta = (assign_cost[i, f_j] + assign_cost[j, f_i]) - \
                            (assign_cost[i, f_i] + assign_cost[j, f_j])
                    if delta < -1e-9:
                        result[i], result[j] = f_j, f_i
                        load[f_i] = load[f_i] - d_i + d_j
                        load[f_j] = load[f_j] - d_j + d_i
                        improved = True
                        break
        return result, load, is_open

    def close_facility_search(result, load, is_open, max_passes=10):
        """2-opt: try closing an open facility, reassign its customers greedily,
        apply if total cost decreases. This is the key improvement over parent."""
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for f_old in range(n_facilities):
                if not _time_ok():
                    return result, load, is_open
                if not is_open[f_old]:
                    continue
                customers_at = np.where(result == f_old)[0]
                if len(customers_at) == 0:
                    continue

                # baseline cost contribution of this facility
                old_open = open_cost[f_old]
                old_assign = float(np.sum(assign_cost[customers_at, f_old]))
                old_total = old_open + old_assign

                # simulate reassignment to other facilities
                temp_load = load.copy()
                temp_load[f_old] = 0.0
                temp_is_open = is_open.copy()
                temp_is_open[f_old] = False

                new_assign = 0.0
                new_open_extra = 0.0
                feasible = True
                plan = {}

                # process customers at f_old by demand desc
                order = customers_at[np.argsort(-demands[customers_at])]
                for i in order:
                    d = demands[i]
                    cand = np.arange(n_facilities)
                    cand = cand[cand != f_old]
                    cand = cand[temp_load[cand] + d <= capacities[cand] + 1e-9]
                    if len(cand) == 0:
                        feasible = False
                        break
                    marg = assign_cost[i, cand]
                    marg[~temp_is_open[cand]] += open_cost[~temp_is_open[cand]]
                    f_new = int(cand[np.argmin(marg)])
                    plan[i] = f_new
                    new_assign += assign_cost[i, f_new]
                    if not temp_is_open[f_new]:
                        new_open_extra += open_cost[f_new]
                        temp_is_open[f_new] = True
                    temp_load[f_new] += d

                if not feasible:
                    continue

                new_total = new_assign + new_open_extra
                if new_total < old_total - 1e-9:
                    # apply
                    for i, f_new in plan.items():
                        result[i] = f_new
                        load[f_new] += demands[i]
                        if not is_open[f_new]:
                            is_open[f_new] = True
                    load[f_old] = 0.0
                    is_open[f_old] = False
                    improved = True
        return result, load, is_open

    def full_local_search(result, load, is_open):
        # alternate the three moves until no improvement
        prev = total_cost(result)
        for _ in range(30):
            if not _time_ok():
                break
            result, load, is_open = relocation_search(result, load, is_open)
            result, load, is_open = swap_search(result, load, is_open)
            result, load, is_open = close_facility_search(result, load, is_open)
            cur = total_cost(result)
            if cur >= prev - 1e-9:
                break
            prev = cur
        return result, load, is_open

    best_result = None
    best_score = INF

    def try_solution(result, load, is_open):
        nonlocal best_result, best_score
        result, load, is_open = full_local_search(result, load, is_open)
        s = total_cost(result)
        if s < best_score:
            best_score = s
            best_result = result.copy()

    # 1) deterministic greedy
    res, load, is_open = greedy_init(seed=None)
    try_solution(res, load, is_open)

    # 2) randomized restarts with optional Lagrangian-style penalty diversification
    n_restarts = 200
    for k in range(n_restarts):
        if not _time_ok():
            break
        # diversify: penalty proportional to a random facility load bias
        if k % 4 == 0:
            # no penalty
            pen = None
        else:
            # random penalty to encourage different facility sets
            rng = np.random.RandomState(1000 + k)
            pen = rng.rand(n_facilities) * 10.0

        res, load, is_open = greedy_init(seed=k, penalty=pen)
        try_solution(res, load, is_open)

    if best_result is None:
        # fallback: all to first feasible
        best_result = np.zeros(n_customers, dtype=int)

    return best_result
