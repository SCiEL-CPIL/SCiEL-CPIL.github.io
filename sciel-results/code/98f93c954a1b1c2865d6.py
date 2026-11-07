"""
Facility location solver: MIP via scipy.optimize.milp with strong local search
including 2-swap moves, plus a greedy fallback.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility.
    Primary: MIP via scipy.optimize.milp (HiGHS), then local search with 2-swaps.
    Fallback: greedy + local search.
    """
    start_time = time.time()
    deadline = start_time + 140.0

    n_customers = len(demands)
    n_facilities = len(capacities)

    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    def compute_cost(assignment):
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            f = int(assignment[c])
            if f < 0 or f >= n_facilities:
                return np.inf
            load[f] += demands[c]
        for f in range(n_facilities):
            if load[f] > capacities[f] + 1e-9:
                return np.inf
        cost = 0.0
        used = np.zeros(n_facilities, dtype=bool)
        for c in range(n_customers):
            f = int(assignment[c])
            cost += assign_cost[c, f]
            used[f] = True
        for f in range(n_facilities):
            if used[f]:
                cost += open_cost[f]
        return cost

    def greedy_assignment(order=None):
        if order is None:
            order = np.arange(n_customers)
        result = np.full(n_customers, -1, dtype=int)
        load = np.zeros(n_facilities)
        for c in order:
            costs = assign_cost[c].copy()
            feasible = load + demands[c] <= capacities
            costs[~feasible] = np.inf
            f = int(np.argmin(costs))
            if feasible[f]:
                result[c] = f
                load[f] += demands[c]
            else:
                return None
        return result

    def local_search(assignment, max_iter=300):
        """Improve via single reassignments and 2-swap moves."""
        assignment = assignment.copy()
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            load[int(assignment[c])] += demands[c]
        current_cost = compute_cost(assignment)
        if current_cost == np.inf:
            return assignment

        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            improved = False

            # --- Single-move phase ---
            for c in range(n_customers):
                old_f = int(assignment[c])
                load[old_f] -= demands[c]
                best_new_f = old_f
                best_delta = 0.0
                for new_f in range(n_facilities):
                    if new_f == old_f:
                        continue
                    if load[new_f] + demands[c] > capacities[new_f] + 1e-9:
                        continue
                    delta = assign_cost[c, new_f] - assign_cost[c, old_f]
                    if load[new_f] == 0:
                        delta += open_cost[new_f]
                    if load[old_f] == 0:
                        delta -= open_cost[old_f]
                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_new_f = new_f
                if best_new_f != old_f and best_delta < -1e-9:
                    assignment[c] = best_new_f
                    load[best_new_f] += demands[c]
                    current_cost += best_delta
                    improved = True
                else:
                    load[old_f] += demands[c]

            # --- 2-swap phase: try swapping facilities of pairs (c1, c2) ---
            # To keep it tractable, sample a subset of pairs
            if time.time() > deadline - 15:
                break
            n_swaps_to_try = min(2000, n_customers * (n_customers - 1) // 2)
            # Use deterministic sampling based on iteration
            rng = np.random.RandomState(it * 7919 + 13)
            tried = set()
            swap_improved = False
            for _ in range(n_swaps_to_try):
                if time.time() > deadline - 10:
                    break
                c1 = rng.randint(n_customers)
                c2 = rng.randint(n_customers)
                if c1 == c2 or (c1, c2) in tried:
                    continue
                tried.add((c1, c2))
                f1 = int(assignment[c1])
                f2 = int(assignment[c2])
                if f1 == f2:
                    continue
                # Check feasibility of swapping: c1->f2, c2->f1
                # Temporarily remove both
                load[f1] -= demands[c1]
                load[f2] -= demands[c2]
                if load[f1] + demands[c2] > capacities[f1] + 1e-9:
                    load[f1] += demands[c1]
                    load[f2] += demands[c2]
                    continue
                if load[f2] + demands[c1] > capacities[f2] + 1e-9:
                    load[f1] += demands[c1]
                    load[f2] += demands[c2]
                    continue
                # Compute delta
                delta = (assign_cost[c1, f2] - assign_cost[c1, f1]) + \
                        (assign_cost[c2, f1] - assign_cost[c2, f2])
                # Open cost changes
                if load[f1] == 0:
                    # f1 becomes empty -> save open_cost[f1]
                    delta -= open_cost[f1]
                if load[f2] == 0:
                    # f2 becomes empty -> save open_cost[f2]
                    delta -= open_cost[f2]
                # After swap, f1 gets c2 (demand>0) so it's open; f2 gets c1 so it's open
                # But if they were already open (load>0 before removal), no change
                # We already accounted for load becoming 0 above.
                # If load[f1] was >0 before removal, it stays >0 (since demands[c2]>0) -> no open cost change
                # Same for f2.
                if delta < -1e-9:
                    assignment[c1] = f2
                    assignment[c2] = f1
                    load[f1] += demands[c2]
                    load[f2] += demands[c1]
                    current_cost += delta
                    improved = True
                    swap_improved = True
                else:
                    load[f1] += demands[c1]
                    load[f2] += demands[c2]

            if not improved and not swap_improved:
                break
            if not improved:
                # Only swap improved, continue to allow more swaps
                pass
            # If neither improved, break
            if not improved and not swap_improved:
                break

        return assignment

    # --- Try MIP ---
    best_assignment = None
    best_cost = np.inf

    try:
        from scipy.optimize import milp, LinearConstraint, Bounds

        n_vars = n_customers * n_facilities + n_facilities
        x_vars = n_customers * n_facilities

        c_obj = np.zeros(n_vars)
        for c in range(n_customers):
            for f in range(n_facilities):
                c_obj[c * n_facilities + f] = assign_cost[c, f]
        for f in range(n_facilities):
            c_obj[x_vars + f] = open_cost[f]

        # Build constraints sparsely
        from scipy.sparse import csr_matrix

        rows = []
        cols = []
        vals = []
        lb_list = []
        ub_list = []

        # Type 1: assignment equality (n_customers rows)
        for c in range(n_customers):
            for f in range(n_facilities):
                rows.append(c)
                cols.append(c * n_facilities + f)
                vals.append(1.0)
            lb_list.append(1.0)
            ub_list.append(1.0)

        # Type 2: capacity (n_facilities rows)
        for f in range(n_facilities):
            row_idx = n_customers + f
            for c in range(n_customers):
                rows.append(row_idx)
                cols.append(c * n_facilities + f)
                vals.append(demands[c])
            rows.append(row_idx)
            cols.append(x_vars + f)
            vals.append(-capacities[f])
            lb_list.append(-np.inf)
            ub_list.append(0.0)

        # Type 3: openness (n_customers * n_facilities rows)
        for c in range(n_customers):
            for f in range(n_facilities):
                row_idx = n_customers + n_facilities + c * n_facilities + f
                rows.append(row_idx)
                cols.append(c * n_facilities + f)
                vals.append(1.0)
                rows.append(row_idx)
                cols.append(x_vars + f)
                vals.append(-1.0)
                lb_list.append(-np.inf)
                ub_list.append(0.0)

        n_cons = len(lb_list)
        A = csr_matrix((vals, (rows, cols)), shape=(n_cons, n_vars))
        lb = np.array(lb_list)
        ub = np.array(ub_list)

        constraints = LinearConstraint(A, lb, ub)
        var_bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
        integrality = np.ones(n_vars)

        result = milp(
            c=c_obj,
            constraints=constraints,
            integrality=integrality,
            bounds=var_bounds,
            options={"time_limit": 100}
        )

        if result.status == 0 and result.x is not None:
            x_sol = result.x[:x_vars].reshape(n_customers, n_facilities)
            assignment = np.argmax(x_sol, axis=1).astype(int)
            # Verify feasibility
            if compute_cost(assignment) < np.inf:
                best_assignment = assignment
                best_cost = compute_cost(assignment)
            else:
                # Try rounding more carefully
                for c in range(n_customers):
                    best_f = 0
                    for f in range(n_facilities):
                        if x_sol[c, f] > x_sol[c, best_f]:
                            best_f = f
                    assignment[c] = best_f
                if compute_cost(assignment) < np.inf:
                    best_assignment = assignment
                    best_cost = compute_cost(assignment)

    except Exception:
        pass

    # --- Fallback or supplement: greedy ---
    if best_assignment is None or time.time() < deadline - 30:
        greedy = greedy_assignment()
        if greedy is not None:
            gc = compute_cost(greedy)
            if gc < best_cost:
                best_assignment = greedy
                best_cost = gc

    # If still nothing, try greedy with different orders
    if best_assignment is None:
        # Sort by demand descending
        order = np.argsort(-demands)
        greedy = greedy_assignment(order)
        if greedy is not None:
            best_assignment = greedy
            best_cost = compute_cost(greedy)

    if best_assignment is None:
        # Last resort: simple assignment
        best_assignment = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            best_f = 0
            best_rem = -1
            for f in range(n_facilities):
                rem = capacities[f] - load[f]
                if rem >= demands[c] and rem > best_rem:
                    best_rem = rem
                    best_f = f
            best_assignment[c] = best_f
            load[best_f] += demands[c]

    # --- Local search polish ---
    if time.time() < deadline - 15:
        best_assignment = local_search(best_assignment, max_iter=200)
        final_cost = compute_cost(best_assignment)
        if final_cost < best_cost:
            best_cost = final_cost

    return best_assignment.astype(int)
# EVOLVE-BLOCK-END
