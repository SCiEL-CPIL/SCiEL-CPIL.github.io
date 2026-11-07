"""Capacitated facility location: LP-guided rounding + vectorized local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linprog

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    n_customers = len(demands)
    n_facilities = len(capacities)

    # ---------- Greedy fallback ----------
    def greedy_assignment():
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities, dtype=np.float64)
        order = np.argsort(-demands)
        for i in order:
            best_f = -1
            best_cost = np.inf
            for f in range(n_facilities):
                if load[f] + demands[i] <= capacities[f]:
                    c = assign_cost[i, f]
                    if c < best_cost:
                        best_cost = c
                        best_f = f
            if best_f == -1:
                best_f = 0
            result[i] = best_f
            load[best_f] += demands[i]
        return result, load

    def total_cost(result):
        cost = 0.0
        used = set(result.tolist())
        for i in range(n_customers):
            cost += assign_cost[i, result[i]]
        for f in used:
            cost += open_cost[f]
        return cost

    # ---------- LP relaxation ----------
    def solve_lp():
        """
        LP relaxation of CFLP:
        Variables: x[i,f] in [0,1] for each customer i, facility f
                  y[f] in [0,1] for each facility f
        Minimize: sum_{i,f} assign_cost[i,f]*x[i,f] + sum_f open_cost[f]*y[f]
        Subject to:
                  sum_f x[i,f] = 1 for all i
                  sum_i demands[i]*x[i,f] <= capacities[f]*y[f] for all f
                  x[i,f] <= y[f] for all i,f  (only if facility open)
                  0 <= x[i,f] <= 1, 0 <= y[f] <= 1
        """
        n_vars = n_customers * n_facilities + n_facilities
        x_offset = 0
        y_offset = n_customers * n_facilities

        # Objective
        c = np.zeros(n_vars)
        c[:n_customers * n_facilities] = assign_cost.ravel()
        c[y_offset:] = open_cost

        # Equality constraints: sum_f x[i,f] = 1 for each i
        A_eq = np.zeros((n_customers, n_vars))
        for i in range(n_customers):
            A_eq[i, i * n_facilities:(i + 1) * n_facilities] = 1.0
        b_eq = np.ones(n_customers)

        # Inequality constraints:
        # (1) capacity: sum_i demands[i]*x[i,f] <= capacities[f]*y[f]
        #     => sum_i demands[i]*x[i,f] - capacities[f]*y[f] <= 0
        # (2) linking: x[i,f] - y[f] <= 0
        n_ineq = n_facilities + n_customers * n_facilities
        A_ub = np.zeros((n_ineq, n_vars))
        b_ub = np.zeros(n_ineq)

        # Capacity constraints
        for f in range(n_facilities):
            row = f
            for i in range(n_customers):
                A_ub[row, i * n_facilities + f] = demands[i]
            A_ub[row, y_offset + f] = -capacities[f]
            b_ub[row] = 0.0

        # Linking constraints
        row = n_facilities
        for i in range(n_customers):
            for f in range(n_facilities):
                A_ub[row, i * n_facilities + f] = 1.0
                A_ub[row, y_offset + f] = -1.0
                b_ub[row] = 0.0
                row += 1

        bounds = [(0.0, 1.0)] * n_vars

        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds, method='highs')
            if res.success:
                x = res.x[:n_customers * n_facilities].reshape(n_customers, n_facilities)
                y = res.x[y_offset:]
                return x, y
        except Exception:
            pass
        return None, None

    # ---------- LP-guided rounding ----------
    def lp_rounding(x_lp, y_lp):
        """
        Deterministic rounding:
        1. Sort customers by decreasing LP "confidence" (max x[i,f]).
        2. Assign each customer to the facility with highest x[i,f] that has capacity.
        3. If no capacity, assign to facility with min marginal cost that has capacity.
        4. Open facilities as needed.
        """
        result = np.full(n_customers, -1, dtype=int)
        load = np.zeros(n_facilities, dtype=np.float64)
        used = set()

        # Priority: customers with higher max LP value first (more "decided")
        max_lp = x_lp.max(axis=1)
        order = np.argsort(-max_lp)

        for i in order:
            # Try facilities sorted by LP value descending
            f_order = np.argsort(-x_lp[i])
            assigned = False
            for f in f_order:
                if load[f] + demands[i] <= capacities[f]:
                    result[i] = f
                    load[f] += demands[i]
                    used.add(f)
                    assigned = True
                    break
            if not assigned:
                # Fallback: cheapest feasible
                for f in range(n_facilities):
                    if load[f] + demands[i] <= capacities[f]:
                        result[i] = f
                        load[f] += demands[i]
                        used.add(f)
                        assigned = True
                        break
            if not assigned:
                # Force into facility 0 (shouldn't happen if capacities are sufficient)
                result[i] = 0
                load[0] += demands[i]
                used.add(0)

        return result, load, used

    # ---------- Vectorized 1-swap local search ----------
    def local_search_1swap(result, load, used, current_cost):
        """Iterate until no improving 1-move found. Vectorized inner loop."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n_customers):
                old_f = result[i]
                # Compute delta for all facilities at once
                delta_assign = assign_cost[i] - assign_cost[i, old_f]

                # Capacity feasibility: load[f] + demands[i] <= capacities[f]
                feasible = (load + demands[i] <= capacities)
                feasible[old_f] = False  # skip same facility

                # Open cost delta
                # If old_f becomes empty: -open_cost[old_f]
                old_empty = (load[old_f] == demands[i])
                # If new_f was empty: +open_cost[new_f]
                new_empty = (load == 0)

                delta_open = np.zeros(n_facilities)
                if old_empty:
                    delta_open[:] = -open_cost[old_f]
                # For facilities that are currently empty, add open_cost
                delta_open[new_empty] += open_cost[new_empty]

                delta = delta_assign + delta_open
                delta[~feasible] = np.inf
                delta[old_f] = np.inf

                best_idx = np.argmin(delta)
                if delta[best_idx] < -1e-9:
                    # Apply move
                    load[old_f] -= demands[i]
                    if load[old_f] == 0:
                        used.discard(old_f)
                    load[best_idx] += demands[i]
                    if best_idx not in used:
                        used.add(best_idx)
                    result[i] = best_idx
                    current_cost += delta[best_idx]
                    improved = True
                    break  # restart scan from customer 0
        return result, load, used, current_cost

    # ---------- 2-swap local search ----------
    def local_search_2swap(result, load, used, current_cost):
        """Try swapping pairs of customers between facilities."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n_customers):
                if time.time() > deadline:
                    break
                fi = result[i]
                for j in range(i + 1, n_customers):
                    fj = result[j]
                    if fi == fj:
                        continue
                    # Swap: i->fj, j->fi
                    # Check capacity
                    if load[fj] - demands[j] + demands[i] > capacities[fj]:
                        continue
                    if load[fi] - demands[i] + demands[j] > capacities[fi]:
                        continue

                    delta_assign = (assign_cost[i, fj] - assign_cost[i, fi]) + \
                                   (assign_cost[j, fi] - assign_cost[j, fj])

                    # Open cost changes
                    delta_open = 0.0
                    # After swap, check if fi or fj become empty
                    new_load_fi = load[fi] - demands[i] + demands[j]
                    new_load_fj = load[fj] - demands[j] + demands[i]

                    # If fi was only serving i (and j is moving in, so not empty unless demands match)
                    # Actually need to check: was fi empty after removing i and before adding j?
                    # fi load after removing i: load[fi] - demands[i]
                    # If that's 0, then adding j makes it non-empty (unless demands[j]=0 which isn't)
                    # So fi won't become empty in a swap (j is moving in)
                    # Similarly fj won't become empty
                    # But what if fi == fj? Already skipped.
                    # The only open-cost change is if a facility was open and becomes empty,
                    # or was closed and becomes open. In a swap, both facilities remain occupied
                    # (since j moves to fi and i moves to fj, and both had at least one customer).
                    # Actually: if fi had only customer i, after swap fi has customer j (still open).
                    # If fj had only customer j, after swap fj has customer i (still open).
                    # So no open cost change in 2-swaps between two occupied facilities.
                    # Unless one of them had only that one customer and the other had only that one,
                    # but they're still occupied after swap.

                    delta = delta_assign + delta_open
                    if delta < -1e-9:
                        # Apply swap
                        load[fi] = new_load_fi
                        load[fj] = new_load_fj
                        result[i] = fj
                        result[j] = fi
                        current_cost += delta
                        improved = True
                        break
                if improved:
                    break
        return result, load, used, current_cost

    # ---------- Main algorithm ----------
    # Try LP relaxation
    x_lp, y_lp = solve_lp()

    if x_lp is not None:
        result, load, used = lp_rounding(x_lp, y_lp)
    else:
        result, load = greedy_assignment()
        used = set(result.tolist())

    current_cost = total_cost(result)

    # 1-swap local search
    result, load, used, current_cost = local_search_1swap(result, load, used, current_cost)

    # 2-swap local search
    result, load, used, current_cost = local_search_2swap(result, load, used, current_cost)

    # Another round of 1-swap after 2-swaps
    result, load, used, current_cost = local_search_1swap(result, load, used, current_cost)

    # Final 2-swap pass
    result, load, used, current_cost = local_search_2swap(result, load, used, current_cost)

    # Verify validity
    load_check = np.zeros(n_facilities, dtype=np.float64)
    valid = True
    for i in range(n_customers):
        f = int(result[i])
        if f < 0 or f >= n_facilities:
            valid = False
            break
        load_check[f] += demands[i]

    if not valid or np.any(load_check > capacities + 1e-6):
        result, load = greedy_assignment()
        current_cost = total_cost(result)
        # Quick 1-swap on fallback
        used = set(result.tolist())
        result, load, used, current_cost = local_search_1swap(result, load, used, current_cost)

    result = result.astype(int)
    return result
# EVOLVE-BLOCK-END
