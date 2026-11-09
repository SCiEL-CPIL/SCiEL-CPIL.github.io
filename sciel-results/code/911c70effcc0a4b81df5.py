"""Capacitated facility location: MILP + greedy/local-search hybrid."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility, respecting capacities, minimizing
    open cost + assignment cost.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit

    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    n_cust = len(demands)
    n_fac = len(capacities)

    def compute_score(res):
        load = np.zeros(n_fac)
        np.add.at(load, res, demands)
        if np.any(load > capacities + 1e-9):
            return np.inf
        used = load > 0
        return float(np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(n_cust), res]))

    def greedy_init():
        """Greedy: assign largest demands first to cheapest feasible facility."""
        res = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        order = np.argsort(-demands)
        for i in order:
            fac_order = np.argsort(assign_cost[i])
            assigned = False
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    res[i] = f
                    load[f] += demands[i]
                    assigned = True
                    break
            if not assigned:
                # Fallback: facility with most remaining capacity
                space = capacities - load
                best_f = int(np.argmax(space))
                if space[best_f] >= demands[i]:
                    res[i] = best_f
                    load[best_f] += demands[i]
                else:
                    res[i] = int(fac_order[0])
        return res

    def local_search(res, load, max_iter=2000):
        """Local search with single-customer relocate moves."""
        improved = True
        it = 0
        while improved and it < max_iter:
            if time.time() > deadline:
                break
            improved = False
            it += 1
            for i in np.random.permutation(n_cust):
                current_f = res[i]
                best_f = -1
                best_delta = 1e-9
                for f_new in range(n_fac):
                    if f_new == current_f:
                        continue
                    if load[f_new] + demands[i] > capacities[f_new] + 1e-9:
                        continue
                    delta = assign_cost[i, f_new] - assign_cost[i, current_f]
                    if load[f_new] == 0:
                        delta += open_cost[f_new]
                    if load[current_f] == demands[i]:
                        delta -= open_cost[current_f]
                    if delta < best_delta:
                        best_delta = delta
                        best_f = f_new
                if best_f != -1:
                    load[current_f] -= demands[i]
                    load[best_f] += demands[i]
                    res[i] = best_f
                    improved = True
        return res, compute_score(res)

    # --- Phase 1: Greedy + local search for a good initial solution ---
    res_greedy = greedy_init()
    load_greedy = np.zeros(n_fac)
    np.add.at(load_greedy, res_greedy, demands)
    local_search(res_greedy, load_greedy, max_iter=500)
    best_assign = res_greedy.copy()
    best_score = compute_score(best_assign)

    # --- Phase 2: MILP for exact/optimal solution ---
    n_x = n_cust * n_fac
    n_y = n_fac
    n_vars = n_x + n_y

    c = np.zeros(n_vars)
    c[:n_x] = assign_cost.flatten()
    c[n_x:] = open_cost

    integrality = np.ones(n_vars)
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)

    # Build constraint matrix
    # Constraints:
    # 1. Each customer assigned to exactly one facility: n_cust constraints
    # 2. Capacity: sum_i d_i x[i,j] <= cap[j] * y[j] for each j: n_fac constraints
    #    (This implicitly enforces x[i,j] <= y[j] when d_i > 0)
    n_constraints = n_cust + n_fac
    A = np.zeros((n_constraints, n_vars))
    lb_c = np.zeros(n_constraints)
    ub_c = np.zeros(n_constraints)

    # Constraint 1: assignment
    for i in range(n_cust):
        for j in range(n_fac):
            A[i, i * n_fac + j] = 1.0
        lb_c[i] = 1.0
        ub_c[i] = 1.0

    # Constraint 2: capacity with linking
    for j in range(n_fac):
        row = n_cust + j
        for i in range(n_cust):
            A[row, i * n_fac + j] = demands[i]
        A[row, n_x + j] = -capacities[j]
        lb_c[row] = -np.inf
        ub_c[row] = 0.0

    linear_constraint = LinearConstraint(A, lb_c, ub_c)
    bounds = Bounds(lb, ub)

    # Solve MILP with time limit
    milp_time_limit = max(5.0, deadline - time.time() - 5.0)
    try:
        result = milp(
            c=c,
            constraints=linear_constraint,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": milp_time_limit}
        )

        if result.status == 0 and result.x is not None:
            x_sol = result.x[:n_x].reshape(n_cust, n_fac)
            res_milp = np.argmax(x_sol, axis=1).astype(int)

            load_milp = np.zeros(n_fac)
            np.add.at(load_milp, res_milp, demands)

            if not np.any(load_milp > capacities + 1e-9):
                score_milp = compute_score(res_milp)
                if score_milp < best_score:
                    best_score = score_milp
                    best_assign = res_milp.copy()

        # Run local search on the best solution found
        if time.time() < deadline - 3.0:
            load_best = np.zeros(n_fac)
            np.add.at(load_best, best_assign, demands)
            local_search(best_assign, load_best, max_iter=1000)
            new_score = compute_score(best_assign)
            if new_score < best_score:
                best_score = new_score

    except Exception:
        pass

    # Final verification
    load_verify = np.zeros(n_fac)
    np.add.at(load_verify, best_assign, demands)
    if np.any(load_verify > capacities + 1e-9):
        best_assign = greedy_init()

    return best_assign.astype(int)

# EVOLVE-BLOCK-END
