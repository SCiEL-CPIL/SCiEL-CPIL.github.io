# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds


def _compute_cost(result, demands, open_cost, assign_cost):
    """Compute total cost for a given assignment."""
    total_assign = np.sum(assign_cost[np.arange(len(demands)), result])
    open_mask = np.zeros(len(open_cost), dtype=bool)
    for f in result:
        open_mask[f] = True
    total_open = np.sum(open_cost[open_mask])
    return total_assign + total_open


def _repair(result, demands, capacities, assign_cost):
    """Fix capacity violations by moving customers from overloaded facilities to cheapest feasible ones."""
    n = len(demands)
    m = len(capacities)
    load = np.zeros(m)
    for i in range(n):
        load[result[i]] += demands[i]

    for _ in range(500):
        overloaded = np.where(load > capacities + 1e-6)[0]
        if len(overloaded) == 0:
            break
        for f_over in overloaded:
            excess = load[f_over] - capacities[f_over]
            custs_at_f = [i for i in range(n) if result[i] == f_over]
            custs_at_f.sort(key=lambda i: -demands[i])
            for i in custs_at_f:
                if excess <= 1e-6:
                    break
                d = demands[i]
                best_f = -1
                best_cost = np.inf
                for f in range(m):
                    if f == f_over:
                        continue
                    if load[f] + d <= capacities[f] + 1e-6:
                        c = assign_cost[i, f]
                        if c < best_cost:
                            best_cost = c
                            best_f = f
                if best_f >= 0:
                    load[f_over] -= d
                    load[best_f] += d
                    result[i] = best_f
                    excess -= d
                else:
                    rem = capacities - load
                    best_f = int(np.argmax(rem))
                    load[f_over] -= d
                    load[best_f] += d
                    result[i] = best_f
                    excess -= d
    return result


def _greedy_init(demands, capacities, open_cost, assign_cost, order=None):
    """Greedy assignment: process customers, pick cheapest feasible facility."""
    n = len(demands)
    m = len(capacities)
    result = np.full(n, -1, dtype=int)
    load = np.zeros(m)

    if order is None:
        order = np.argsort(-demands)
    for i in order:
        d = demands[i]
        best_f = -1
        best_cost = np.inf
        for f in range(m):
            if load[f] + d <= capacities[f] + 1e-6:
                c = assign_cost[i, f]
                if c < best_cost:
                    best_cost = c
                    best_f = f
        if best_f >= 0:
            result[i] = best_f
            load[best_f] += d
        else:
            rem = capacities - load
            best_f = int(np.argmax(rem))
            result[i] = best_f
            load[best_f] += d

    return _repair(result, demands, capacities, assign_cost)


def _cheapest_init(demands, capacities, open_cost, assign_cost):
    """Assign each customer to their cheapest facility, then repair capacity violations."""
    n = len(demands)
    m = len(capacities)
    result = np.argmin(assign_cost, axis=1).astype(int)
    return _repair(result, demands, capacities, assign_cost)


def _local_search(result, demands, capacities, open_cost, assign_cost,
                  max_iters=500, deadline=None, rng=None, do_swap=True):
    """Local search with single moves and 2-opt swaps."""
    if rng is None:
        rng = np.random.default_rng()

    n = len(demands)
    m = len(capacities)

    load = np.zeros(m)
    for i in range(n):
        load[result[i]] += demands[i]

    current_cost = _compute_cost(result, demands, open_cost, assign_cost)

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        improved = False
        order = rng.permutation(n)

        # Single moves
        for i in order:
            current_f = result[i]
            d = demands[i]
            best_f = -1
            best_delta = 0.0

            for f in range(m):
                if f == current_f:
                    continue
                if load[f] + d > capacities[f] + 1e-6:
                    continue

                delta = assign_cost[i, f] - assign_cost[i, current_f]

                if load[current_f] == d:  # current_f becomes empty
                    delta -= open_cost[current_f]
                if load[f] == 0:  # f becomes non-empty
                    delta += open_cost[f]

                if delta < best_delta - 1e-9:
                    best_delta = delta
                    best_f = f

            if best_f >= 0:
                load[current_f] -= d
                load[best_f] += d
                result[i] = best_f
                current_cost += best_delta
                improved = True

        # 2-opt swaps
        if do_swap and (deadline is None or time.time() < deadline):
            swap_order = rng.permutation(n)
            for i in swap_order:
                if deadline is not None and time.time() > deadline:
                    break
                fi = result[i]
                di = demands[i]
                for j in range(n):
                    if j == i:
                        continue
                    fj = result[j]
                    if fj == fi:
                        continue
                    dj = demands[j]

                    new_load_fi = load[fi] - di + dj
                    new_load_fj = load[fj] - dj + di
                    if new_load_fi > capacities[fi] + 1e-6:
                        continue
                    if new_load_fj > capacities[fj] + 1e-6:
                        continue

                    delta = (assign_cost[i, fj] + assign_cost[j, fi]) - \
                            (assign_cost[i, fi] + assign_cost[j, fj])

                    # open cost: a facility changes open status only if it goes to/from zero
                    if load[fi] > 0 and new_load_fi == 0:
                        delta -= open_cost[fi]
                    elif load[fi] == 0 and new_load_fi > 0:
                        delta += open_cost[fi]
                    if load[fj] > 0 and new_load_fj == 0:
                        delta -= open_cost[fj]
                    elif load[fj] == 0 and new_load_fj > 0:
                        delta += open_cost[fj]

                    if delta < -1e-9:
                        load[fi] = new_load_fi
                        load[fj] = new_load_fj
                        result[i] = fj
                        result[j] = fi
                        current_cost += delta
                        improved = True

        if not improved:
            break

    return result, current_cost


def _solve_lp_flow(demands, capacities, open_cost, assign_cost, deadline):
    """LP relaxation via big-M min-cost-flow style formulation. Returns integer assignment or None."""
    n = len(demands)
    m = len(capacities)

    # Variables: x[i,f] (n*m), y[f] (m)
    total_vars = n * m + m

    # Big M: open cost is charged via y; use big-M on arcs so LP pushes y to 0/1
    # Objective: sum assign_cost[i,f]*x[i,f] + sum open_cost[f]*y[f]
    c = np.zeros(total_vars)
    for i in range(n):
        for f in range(m):
            c[i * m + f] = assign_cost[i, f]
    for f in range(m):
        c[n * m + f] = open_cost[f]

    # Constraints
    A_rows = []
    lb = []
    ub = []

    # Assignment: sum_f x[i,f] = 1
    for i in range(n):
        row = np.zeros(total_vars)
        for f in range(m):
            row[i * m + f] = 1.0
        A_rows.append(row)
        lb.append(1.0)
        ub.append(1.0)

    # Capacity: sum_i demands[i]*x[i,f] <= capacities[f]
    for f in range(m):
        row = np.zeros(total_vars)
        for i in range(n):
            row[i * m + f] = demands[i]
        A_rows.append(row)
        lb.append(0.0)
        ub.append(capacities[f])

    # Linking: x[i,f] - y[f] <= 0
    for i in range(n):
        for f in range(m):
            row = np.zeros(total_vars)
            row[i * m + f] = 1.0
            row[n * m + f] = -1.0
            A_rows.append(row)
            lb.append(-np.inf)
            ub.append(0.0)

    A = np.array(A_rows)
    constraint = LinearConstraint(A, lb=np.array(lb), ub=np.array(ub))
    bounds = Bounds(lb=np.zeros(total_vars), ub=np.ones(total_vars))

    remaining_time = deadline - time.time()
    if remaining_time <= 0.2:
        return None

    from scipy.optimize import linprog
    res = linprog(
        c=c,
        A_ub=None,
        b_ub=None,
        A_eq=None,
        b_eq=None,
        bounds=bounds,
        method='highs',
        options={'time_limit': max(0.1, remaining_time), 'presolve': True}
    )

    # linprog doesn't accept LinearConstraint; rebuild using A_ub/A_eq form
    return None


def _solve_lp_flow2(demands, capacities, open_cost, assign_cost, deadline):
    """LP relaxation using A_eq / A_ub form with HiGHS."""
    n = len(demands)
    m = len(capacities)
    total_vars = n * m + m

    c = np.zeros(total_vars)
    for i in range(n):
        for f in range(m):
            c[i * m + f] = assign_cost[i, f]
    for f in range(m):
        c[n * m + f] = open_cost[f]

    # Equality: assignment constraints (n)
    A_eq = np.zeros((n, total_vars))
    b_eq = np.ones(n)
    for i in range(n):
        for f in range(m):
            A_eq[i, i * m + f] = 1.0

    # Inequality: capacity (m) and linking (n*m)
    n_ineq = m + n * m
    A_ub = np.zeros((n_ineq, total_vars))
    b_ub = np.zeros(n_ineq)
    for f in range(m):
        for i in range(n):
            A_ub[f, i * m + f] = demands[i]
        b_ub[f] = capacities[f]
    idx = m
    for i in range(n):
        for f in range(m):
            A_ub[idx, i * m + f] = 1.0
            A_ub[idx, n * m + f] = -1.0
            b_ub[idx] = 0.0
            idx += 1

    bounds = [(0.0, 1.0)] * total_vars

    remaining_time = deadline - time.time()
    if remaining_time <= 0.2:
        return None

    from scipy.optimize import linprog
    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs',
                  options={'time_limit': max(0.1, remaining_time)})

    if res.status != 0:
        return None

    x_flat = res.x[:n * m]
    x = x_flat.reshape(n, m)
    # Round: assign each customer to argmax x (LP should be near integral on these)
    result = np.argmax(x, axis=1).astype(int)

    # Verify feasibility
    load = np.zeros(m)
    for i in range(n):
        load[result[i]] += demands[i]
    if np.all(load <= capacities + 1e-6):
        return result
    return None


def _solve_mip(demands, capacities, open_cost, assign_cost, deadline):
    """Solve the capacitated facility location problem as a MIP using HiGHS."""
    n = len(demands)
    m = len(capacities)
    total_vars = n * m + m

    c = np.zeros(total_vars)
    for i in range(n):
        for f in range(m):
            c[i * m + f] = assign_cost[i, f]
    for f in range(m):
        c[n * m + f] = open_cost[f]

    A_rows = []
    lb = []
    ub = []

    for i in range(n):
        row = np.zeros(total_vars)
        for f in range(m):
            row[i * m + f] = 1.0
        A_rows.append(row)
        lb.append(1.0)
        ub.append(1.0)

    for f in range(m):
        row = np.zeros(total_vars)
        for i in range(n):
            row[i * m + f] = demands[i]
        A_rows.append(row)
        lb.append(0.0)
        ub.append(capacities[f])

    for i in range(n):
        for f in range(m):
            row = np.zeros(total_vars)
            row[i * m + f] = 1.0
            row[n * m + f] = -1.0
            A_rows.append(row)
            lb.append(-np.inf)
            ub.append(0.0)

    A = np.array(A_rows)
    integrality = np.ones(total_vars)
    bounds = Bounds(lb=np.zeros(total_vars), ub=np.ones(total_vars))
    constraint = LinearConstraint(A, lb=np.array(lb), ub=np.array(ub))

    remaining_time = deadline - time.time()
    if remaining_time <= 0.2:
        return None

    res = milp(c=c, constraints=constraint, integrality=integrality,
               bounds=bounds,
               options={"time_limit": max(0.1, remaining_time),
                        "mip_rel_gap": 0.0})

    if res.status == 0:
        x_flat = res.x[:n * m]
        x = x_flat.reshape(n, m)
        result = np.argmax(x, axis=1).astype(int)
        load = np.zeros(m)
        for i in range(n):
            load[result[i]] += demands[i]
        if np.all(load <= capacities + 1e-6):
            return result
    return None


def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    n = len(demands)
    m = len(capacities)

    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    rng = np.random.default_rng()

    best_result = None
    best_cost = np.inf

    def _update(res):
        nonlocal best_result, best_cost
        if res is None:
            return
        load = np.zeros(m)
        ok = True
        for i in range(n):
            f = int(res[i])
            if f < 0 or f >= m:
                ok = False
                break
            load[f] += demands[i]
        if not ok or np.any(load > capacities + 1e-6):
            return
        cost = _compute_cost(res, demands, open_cost, assign_cost)
        if cost < best_cost:
            best_cost = cost
            best_result = res.copy()

    # 1) LP relaxation warm start (fast, near-integral)
    lp_res = _solve_lp_flow2(demands, capacities, open_cost, assign_cost, deadline)
    if lp_res is not None:
        lp_res, _ = _local_search(lp_res, demands, capacities, open_cost, assign_cost,
                                  max_iters=150, deadline=deadline, rng=rng, do_swap=True)
        _update(lp_res)

    # 2) Heuristic initial solutions + local search
    for init in (_cheapest_init(demands, capacities, open_cost, assign_cost),
                 _greedy_init(demands, capacities, open_cost, assign_cost)):
        if deadline - time.time() < 3.0:
            break
        ls_res, _ = _local_search(init, demands, capacities, open_cost, assign_cost,
                                  max_iters=300, deadline=deadline, rng=rng, do_swap=True)
        _update(ls_res)

    # 3) MIP (exact) with remaining budget
    if deadline - time.time() > 8.0:
        mip_res = _solve_mip(demands, capacities, open_cost, assign_cost, deadline)
        if mip_res is not None:
            mip_res, _ = _local_search(mip_res, demands, capacities, open_cost, assign_cost,
                                       max_iters=100, deadline=deadline, rng=rng, do_swap=True)
            _update(mip_res)

    # 4) Final polish pass on best result
    if best_result is not None and deadline - time.time() > 2.0:
        best_result, _ = _local_search(best_result, demands, capacities, open_cost, assign_cost,
                                       max_iters=200, deadline=deadline, rng=rng, do_swap=True)
        _update(best_result)

    # Fallback: ensure we always return something valid
    if best_result is None:
        best_result = _greedy_init(demands, capacities, open_cost, assign_cost)
        _update(best_result)
    if best_result is None:
        best_result = np.zeros(n, dtype=int)
        _update(best_result)

    # Final validity check
    load = np.zeros(m)
    for i in range(n):
        f = int(best_result[i])
        if f < 0 or f >= m:
            best_result = _greedy_init(demands, capacities, open_cost, assign_cost)
            break
        load[f] += demands[i]
    if np.any(load > capacities + 1e-6):
        best_result = _greedy_init(demands, capacities, open_cost, assign_cost)

    return best_result.astype(int)
# EVOLVE-BLOCK-END
