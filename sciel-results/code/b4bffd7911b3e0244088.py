"""
Facility location solver: MIP via scipy.optimize.milp, LP relaxation for initial
solution, strong local search with 2-swap and 3-swap moves, plus greedy fallback.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility.
    Primary: LP relaxation for initial solution, MIP via scipy.optimize.milp,
    then local search with 2-swaps and 3-swaps.
    Fallback: greedy + local search.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    n_customers = len(demands)
    n_facilities = len(capacities)

    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    def compute_cost(assignment):
        """Compute total cost for an assignment, return inf if infeasible."""
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
        """Greedy: assign in given order to cheapest feasible facility."""
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
        """Improve via single reassignments, 2-swap, and 3-swap moves."""
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

            # --- Single-move phase (vectorized) ---
            for c in range(n_customers):
                old_f = int(assignment[c])
                load[old_f] -= demands[c]
                best_new_f = old_f
                best_delta = 0.0
                # Vectorized delta computation
                feasible = load + demands[c] <= capacities + 1e-9
                feasible[old_f] = False
                if np.any(feasible):
                    deltas = assign_cost[c, feasible] - assign_cost[c, old_f]
                    # Add open cost for facilities becoming open
                    newly_open = (load[feasible] == 0)
                    deltas[newly_open] += open_cost[feasible][newly_open]
                    # Subtract open cost if old facility becomes empty
                    if load[old_f] == 0:
                        deltas -= open_cost[old_f]
                    idx = np.argmin(deltas)
                    if deltas[idx] < best_delta - 1e-9:
                        best_delta = deltas[idx]
                        best_new_f = int(np.where(feasible)[0][idx])
                if best_new_f != old_f and best_delta < -1e-9:
                    assignment[c] = best_new_f
                    load[best_new_f] += demands[c]
                    current_cost += best_delta
                    improved = True
                else:
                    load[old_f] += demands[c]

            # --- 2-swap phase ---
            if time.time() > deadline - 15:
                break
            n_swaps_to_try = min(3000, n_customers * (n_customers - 1) // 2)
            rng = np.random.RandomState(it * 7919 + 13)
            tried = set()
            swap_improved = False
            for _ in range(n_swaps_to_try):
                if time.time() > deadline - 10:
                    break
                c1 = rng.randint(n_customers)
                c2 = rng.randint(n_customers)
                if c1 == c2 or (min(c1,c2), max(c1,c2)) in tried:
                    continue
                tried.add((min(c1,c2), max(c1,c2)))
                f1 = int(assignment[c1])
                f2 = int(assignment[c2])
                if f1 == f2:
                    continue
                # Check feasibility of swapping
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
                if load[f1] == 0:
                    delta -= open_cost[f1]
                if load[f2] == 0:
                    delta -= open_cost[f2]
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

            # --- 3-swap phase (less frequent) ---
            if it % 5 == 0 and time.time() < deadline - 20:
                n_3swaps = min(500, n_customers * (n_customers - 1) * (n_customers - 2) // 6)
                rng3 = np.random.RandomState(it * 104729 + 7)
                tried3 = set()
                for _ in range(n_3swaps):
                    if time.time() > deadline - 15:
                        break
                    c1 = rng3.randint(n_customers)
                    c2 = rng3.randint(n_customers)
                    c3 = rng3.randint(n_customers)
                    if c1 == c2 or c2 == c3 or c1 == c3:
                        continue
                    key = tuple(sorted([c1, c2, c3]))
                    if key in tried3:
                        continue
                    tried3.add(key)
                    f1 = int(assignment[c1])
                    f2 = int(assignment[c2])
                    f3 = int(assignment[c3])
                    # Try cyclic swap: c1->f2, c2->f3, c3->f1
                    if len(set([f1, f2, f3])) < 2:
                        continue
                    # Remove all three
                    load[f1] -= demands[c1]
                    load[f2] -= demands[c2]
                    load[f3] -= demands[c3]
                    # Check feasibility
                    if load[f2] + demands[c1] > capacities[f2] + 1e-9:
                        load[f1] += demands[c1]
                        load[f2] += demands[c2]
                        load[f3] += demands[c3]
                        continue
                    if load[f3] + demands[c2] > capacities[f3] + 1e-9:
                        load[f1] += demands[c1]
                        load[f2] += demands[c2]
                        load[f3] += demands[c3]
                        continue
                    if load[f1] + demands[c3] > capacities[f1] + 1e-9:
                        load[f1] += demands[c1]
                        load[f2] += demands[c2]
                        load[f3] += demands[c3]
                        continue
                    # Compute delta
                    delta = (assign_cost[c1, f2] - assign_cost[c1, f1]) + \
                            (assign_cost[c2, f3] - assign_cost[c2, f2]) + \
                            (assign_cost[c3, f1] - assign_cost[c3, f3])
                    # Open cost changes
                    for f in [f1, f2, f3]:
                        if load[f] == 0:
                            delta -= open_cost[f]
                    if delta < -1e-9:
                        assignment[c1] = f2
                        assignment[c2] = f3
                        assignment[c3] = f1
                        load[f2] += demands[c1]
                        load[f3] += demands[c2]
                        load[f1] += demands[c3]
                        current_cost += delta
                        improved = True
                    else:
                        load[f1] += demands[c1]
                        load[f2] += demands[c2]
                        load[f3] += demands[c3]

            if not improved:
                break

        return assignment

    # --- Try LP relaxation for initial solution ---
    def lp_relaxation():
        """Solve LP relaxation to get fractional solution for rounding."""
        try:
            from scipy.optimize import linprog
            from scipy.sparse import csr_matrix

            # Variables: x[c,f] for c in range(n_customers), f in range(n_facilities)
            n_vars = n_customers * n_facilities
            c_obj = assign_cost.flatten()

            # Constraints:
            # 1. sum_f x[c,f] = 1 for each c (equality)
            # 2. sum_c demands[c]*x[c,f] <= capacities[f] for each f (inequality)

            A_eq_rows = []
            b_eq = []
            for c in range(n_customers):
                row = np.zeros(n_vars)
                row[c * n_facilities:(c+1) * n_facilities] = 1.0
                A_eq_rows.append(row)
                b_eq.append(1.0)

            A_ub_rows = []
            b_ub = []
            for f in range(n_facilities):
                row = np.zeros(n_vars)
                for c in range(n_customers):
                    row[c * n_facilities + f] = demands[c]
                A_ub_rows.append(row)
                b_ub.append(capacities[f])

            A_eq = np.array(A_eq_rows)
            b_eq = np.array(b_eq)
            A_ub = np.array(A_ub_rows)
            b_ub = np.array(b_ub)

            bounds = [(0, 1)] * n_vars

            result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                           bounds=bounds, method='highs')

            if result.status == 0 and result.x is not None:
                x_sol = result.x.reshape(n_customers, n_facilities)
                # Round: for each customer, pick facility with highest x value
                # but respect capacity constraints
                assignment = np.zeros(n_customers, dtype=int)
                load = np.zeros(n_facilities)
                # Sort customers by their best fractional assignment
                order = list(range(n_customers))
                for c in order:
                    best_f = -1
                    best_score = -1
                    for f in range(n_facilities):
                        if load[f] + demands[c] <= capacities[f] + 1e-9:
                            score = x_sol[c, f] - 0.01 * assign_cost[c, f] / (assign_cost.max() + 1)
                            if score > best_score:
                                best_score = score
                                best_f = f
                    if best_f >= 0:
                        assignment[c] = best_f
                        load[best_f] += demands[c]
                    else:
                        # Fallback to cheapest feasible
                        for f in range(n_facilities):
                            if load[f] + demands[c] <= capacities[f] + 1e-9:
                                assignment[c] = f
                                load[best_f] += demands[c]
                                break
                        else:
                            return None
                if compute_cost(assignment) < np.inf:
                    return assignment
                return None
        except Exception:
            return None

    # --- Try MIP ---
    def try_mip():
        """Try MIP via scipy.optimize.milp."""
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            from scipy.sparse import csr_matrix

            n_vars = n_customers * n_facilities + n_facilities
            x_vars = n_customers * n_facilities

            c_obj = np.zeros(n_vars)
            for c in range(n_customers):
                for f in range(n_facilities):
                    c_obj[c * n_facilities + f] = assign_cost[c, f]
            for f in range(n_facilities):
                c_obj[x_vars + f] = open_cost[f]

            # Build constraints sparsely
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
                options={"time_limit": 80}
            )

            if result.status == 0 and result.x is not None:
                x_sol = result.x[:x_vars].reshape(n_customers, n_facilities)
                assignment = np.argmax(x_sol, axis=1).astype(int)
                if compute_cost(assignment) < np.inf:
                    return assignment
                else:
                    for c in range(n_customers):
                        best_f = 0
                        for f in range(n_facilities):
                            if x_sol[c, f] > x_sol[c, best_f]:
                                best_f = f
                        assignment[c] = best_f
                    if compute_cost(assignment) < np.inf:
                        return assignment
            return None
        except Exception:
            return None

    # --- Main logic ---
    best_assignment = None
    best_cost = np.inf

    # 1. Try LP relaxation for a good initial solution
    if time.time() < deadline - 60:
        lp_sol = lp_relaxation()
        if lp_sol is not None:
            lp_cost = compute_cost(lp_sol)
            if lp_cost < best_cost:
                best_assignment = lp_sol
                best_cost = lp_cost

    # 2. Try MIP
    if time.time() < deadline - 30:
        mip_sol = try_mip()
        if mip_sol is not None:
            mip_cost = compute_cost(mip_sol)
            if mip_cost < best_cost:
                best_assignment = mip_sol
                best_cost = mip_cost

    # 3. Greedy fallback with multiple orders
    if best_assignment is None or time.time() < deadline - 20:
        orders = [
            np.arange(n_customers),
            np.argsort(-demands),
            np.argsort(demands),
            np.argsort(assign_cost.mean(axis=1)),
        ]
        for order in orders:
            if time.time() > deadline - 15:
                break
            greedy = greedy_assignment(order)
            if greedy is not None:
                gc = compute_cost(greedy)
                if gc < best_cost:
                    best_assignment = greedy
                    best_cost = gc

    # 4. Last resort
    if best_assignment is None:
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

    # 5. Local search polish
    if time.time() < deadline - 10:
        best_assignment = local_search(best_assignment, max_iter=200)
        final_cost = compute_cost(best_assignment)
        if final_cost < best_cost:
            best_cost = final_cost

    return best_assignment.astype(int)
# EVOLVE-BLOCK-END
