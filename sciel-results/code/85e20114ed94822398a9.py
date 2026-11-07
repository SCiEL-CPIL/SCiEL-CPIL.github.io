"""Capacitated facility location via Lagrangian relaxation + subgradient + local search."""
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    t0 = time.time()
    deadline = t0 + 135.0  # safety margin under 150s wall-clock

    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    n_cust = len(demands)
    n_fac = len(capacities)

    # Precompute
    best_fac_per_cust = np.argmin(assign_cost, axis=1)
    best_cost_per_cust = assign_cost[np.arange(n_cust), best_fac_per_cust]
    total_demand = demands.sum()
    total_cap = capacities.sum()

    def feasible_solution(assign_vec):
        """Check and repair a candidate assignment."""
        load = np.zeros(n_fac)
        for i in range(n_cust):
            f = assign_vec[i]
            load[f] += demands[i]
        # Repair over-capacity: move customers from over-full to under-full
        for _ in range(200):
            over = np.where(load > capacities + 1e-9)[0]
            if len(over) == 0:
                break
            for f in over:
                excess = load[f] - capacities[f]
                # Find customers at f, sort by how much they'd save by moving
                custs_at_f = np.where(assign_vec == f)[0]
                if len(custs_at_f) == 0:
                    continue
                # Try moving customers with cheapest alternative
                for i in custs_at_f:
                    if excess <= 1e-9:
                        break
                    alts = np.argsort(assign_cost[i])
                    moved = False
                    for af in alts:
                        if af == f:
                            continue
                        if load[af] + demands[i] <= capacities[af] + 1e-9:
                            load[f] -= demands[i]
                            load[af] += demands[i]
                            excess -= demands[i]
                            assign_vec[i] = af
                            moved = True
                            break
                    if not moved:
                        # Forced move to least-loaded feasible
                        feasible = np.where(capacities - load >= demands[i] - 1e-9)[0]
                        if len(feasible) > 0:
                            af = feasible[np.argmin(load[feasible])]
                            load[f] -= demands[i]
                            load[af] += demands[i]
                            assign_vec[i] = af
        # Final validity check
        if np.any(load > capacities + 1e-9):
            return None
        return assign_vec

    def solution_cost(assign_vec):
        if assign_vec is None:
            return np.inf
        open_mask = np.zeros(n_fac, dtype=bool)
        for i in range(n_cust):
            open_mask[assign_vec[i]] = True
        open_sum = open_cost[open_mask].sum()
        assign_sum = assign_cost[np.arange(n_cust), assign_vec].sum()
        return open_sum + assign_sum

    # Lagrangian relaxation: dualize capacity constraints
    # L(lambda) = min over assignments (ignoring capacity) of sum_i assign_cost[i, a_i]
    #            + sum_f open_cost_f * y_f  + sum_f lambda_f * (sum_{i:a_i=f} d_i - cap_f)
    # For fixed lambda, optimal assignment for customer i is argmin_f (assign_cost[i,f] + lambda_f * d_i)
    # and facility f is "open" if any customer assigned to it.
    # We use subgradient method on lambda.

    best_solution = None
    best_cost = np.inf

    # Initialize lambda
    # Heuristic: small positive values
    lam = np.full(n_fac, 1.0)

    # Precompute for speed
    # cost_lambda[i, f] = assign_cost[i, f] + lam[f] * demands[i]
    # We'll recompute each iteration

    # Subgradient method parameters
    max_iters = 3000
    step_mult = 1.0
    # Step size: alpha_k = step_mult * (best_cost - L_k) / ||g_k||^2
    # But best_cost may be inf initially; use a reference.

    # First, get an initial feasible solution via greedy
    def greedy_solve():
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        for i in np.argsort(-demands):
            # Sort facilities by assign_cost[i]
            order = np.argsort(assign_cost[i])
            placed = False
            for f in order:
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    load[f] += demands[i]
                    result[i] = f
                    placed = True
                    break
            if not placed:
                # Should not happen if total_cap >= total_demand
                # Force to least loaded
                f = int(np.argmin(load))
                load[f] += demands[i]
                result[i] = f
        return result

    greedy_sol = greedy_solve()
    greedy_sol = feasible_solution(greedy_sol.copy())
    if greedy_sol is not None:
        best_solution = greedy_sol.copy()
        best_cost = solution_cost(best_solution)

    # Run subgradient
    lam = np.full(n_fac, 0.5)
    # Adaptive step
    step_size = 2.0
    best_L = np.inf
    stall = 0

    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        # Compute adjusted cost
        # adj[i, f] = assign_cost[i, f] + lam[f] * demands[i]
        adj = assign_cost + np.outer(demands, lam)
        # Assignment ignoring capacity
        a = np.argmin(adj, axis=1)
        # Load per facility
        load = np.bincount(a, minlength=n_fac).astype(np.float64) * 0
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[a[i]] += demands[i]
        # Open facilities
        open_mask = np.zeros(n_fac, dtype=bool)
        for i in range(n_cust):
            open_mask[a[i]] = True
        open_sum = open_cost[open_mask].sum()
        assign_sum = assign_cost[np.arange(n_cust), a].sum()
        L = open_sum + assign_sum + lam @ (load - capacities)

        # Subgradient for lambda_f: g_f = load_f - cap_f
        g = load - capacities
        # Update lambda: lambda_f = max(0, lambda_f - step * g_f)
        # Step size: use best known feasible cost as upper bound
        if best_cost < np.inf:
            gap = best_cost - L
            if gap < 0:
                gap = 0.0
            norm_g = np.linalg.norm(g)
            if norm_g > 1e-12:
                step = step_mult * gap / (norm_g ** 2)
                step = min(step, step_size)
            else:
                step = 0.0
        else:
            step = step_size
        lam = np.maximum(0.0, lam - step * g)

        # Track best L
        if L < best_L:
            best_L = L
            stall = 0
        else:
            stall += 1
            step_mult *= 0.95

        # Try to get a feasible solution from this assignment via repair
        if it % 5 == 0 or stall % 20 == 0:
            cand = a.copy()
            cand = feasible_solution(cand)
            if cand is not None:
                c = solution_cost(cand)
                if c < best_cost:
                    best_cost = c
                    best_solution = cand.copy()

    # Local search: 1-move and 2-swap
    if best_solution is not None and time.time() < deadline - 10.0:
        sol = best_solution.copy()
        cur_cost = best_cost
        improved = True
        max_ls_iters = 50
        for ls_iter in range(max_ls_iters):
            if time.time() > deadline - 10.0:
                break
            if not improved:
                break
            improved = False
            load = np.zeros(n_fac)
            for i in range(n_cust):
                load[sol[i]] += demands[i]
            open_mask = np.zeros(n_fac, dtype=bool)
            for i in range(n_cust):
                open_mask[sol[i]] = True
            open_sum = open_cost[open_mask].sum()
            # Try 1-move: move customer i to facility f
            # Delta = assign_cost[i,f] - assign_cost[i, sol[i]]
            #        + open_cost change if f becomes newly open or sol[i] becomes empty
            # We'll do a limited pass
            for i in range(n_cust):
                if time.time() > deadline - 10.0:
                    break
                cur_f = sol[i]
                # Candidates: top few cheapest facilities
                order = np.argsort(assign_cost[i])
                for f in order[:8]:
                    if f == cur_f:
                        continue
                    if load[f] + demands[i] > capacities[f] + 1e-9:
                        continue
                    # Compute delta
                    delta = assign_cost[i, f] - assign_cost[i, cur_f]
                    # Open cost change
                    if not open_mask[f]:
                        delta += open_cost[f]
                    # After moving, check if cur_f becomes empty
                    # Count how many at cur_f
                    # For efficiency, we can approximate; but let's be careful
                    # We'll do a full recompute only if delta < 0
                    if delta >= 0:
                        continue
                    # Tentative move
                    sol[i] = f
                    # Check validity
                    load2 = load.copy()
                    load2[cur_f] -= demands[i]
                    load2[f] += demands[i]
                    if np.any(load2 > capacities + 1e-9):
                        sol[i] = cur_f
                        continue
                    # Recompute open cost
                    open_mask2 = open_mask.copy()
                    open_mask2[f] = True
                    # Check if cur_f becomes empty
                    cnt_cur = 0
                    for j in range(n_cust):
                        if sol[j] == cur_f:
                            cnt_cur += 1
                    if cnt_cur == 0:
                        open_mask2[cur_f] = False
                    new_open = open_cost[open_mask2].sum()
                    new_assign = assign_cost[np.arange(n_cust), sol].sum()
                    new_total = new_open + new_assign
                    if new_total < cur_cost - 1e-9:
                        cur_cost = new_total
                        load = load2
                        open_mask = open_mask2
                        improved = True
                        break
                    else:
                        sol[i] = cur_f
        if cur_cost < best_cost:
            best_cost = cur_cost
            best_solution = sol.copy()

    # Final fallback
    if best_solution is None:
        best_solution = greedy_sol
        if best_solution is None:
            # Absolute fallback: round-robin
            best_solution = np.arange(n_cust) % n_fac
            # Repair
            best_solution = feasible_solution(best_solution)
            if best_solution is None:
                # Force valid by capacity
                best_solution = np.zeros(n_cust, dtype=int)
                load = np.zeros(n_fac)
                for i in range(n_cust):
                    for f in range(n_fac):
                        if load[f] + demands[i] <= capacities[f]:
                            load[f] += demands[i]
                            best_solution[i] = f
                            break
    # Ensure integer array
    return best_solution.astype(int)
