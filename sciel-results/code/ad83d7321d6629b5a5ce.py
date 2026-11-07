"""Rewritten with Lagrangian relaxation + subgradient optimization."""
# EVOLVE-BLOCK-START
import numpy as np


def _subgradient(demands, capacities, open_cost, assign_cost,
                 n_restarts=4, max_iter=600, rng=None):
    n = len(demands)
    m = len(capacities)
    if rng is None:
        rng = np.random.default_rng()

    best_sol = None
    best_val = np.inf

    for restart in range(n_restarts):
        # Random non-negative initial multipliers, scaled by typical cost
        scale = float(np.median(assign_cost)) if assign_cost.size else 1.0
        lam = rng.uniform(0.0, scale, size=m)
        step_base = scale / (restart + 1)  # decreasing step schedule

        # Track best feasible assignment seen during this restart
        restart_best = None
        restart_best_val = np.inf

        for it in range(max_iter):
            # Lagrangian assignment: each customer picks cheapest adjusted facility
            adj = assign_cost + lam[None, :]
            fac = np.argmin(adj, axis=1)
            load = np.bincount(fac, weights=demands, minlength=m)

            # Check feasibility
            feasible = np.all(load <= capacities + 1e-9)
            if feasible:
                # Objective = open cost of used facilities + assignment cost
                used = np.unique(fac)
                obj = float(open_cost[used].sum() + assign_cost[np.arange(n), fac].sum())
                if obj < restart_best_val:
                    restart_best_val = obj
                    restart_best = fac.copy()

            # Subgradient update for capacity constraints
            grad = load - capacities
            step = step_base / (1.0 + it * 0.01)
            lam = np.maximum(0.0, lam + step * grad)

        # End of restart: repair any infeasible assignment
        if restart_best is not None:
            cand = restart_best
        else:
            # Take last Lagrangian assignment and repair
            adj = assign_cost + lam[None, :]
            cand = np.argmin(adj, axis=1)
            load = np.bincount(cand, weights=demands, minlength=m)
            # Greedy repair: reassign overloaded customers to cheapest feasible
            order = np.argsort(-demands)
            for i in order:
                if load[cand[i]] <= capacities[cand[i]] + 1e-9:
                    continue
                # find cheapest facility with room
                for f in np.argsort(assign_cost[i]):
                    if load[f] + demands[i] <= capacities[f] + 1e-9:
                        load[cand[i]] -= demands[i]
                        load[f] += demands[i]
                        cand[i] = f
                        break
                else:
                    # No feasible facility found; leave as is (will be invalid, skip)
                    pass

        # Validate and keep best
        if cand is not None:
            load = np.bincount(cand, weights=demands, minlength=m)
            if np.all(load <= capacities + 1e-9):
                used = np.unique(cand)
                obj = float(open_cost[used].sum() + assign_cost[np.arange(n), cand].sum())
                if obj < best_val:
                    best_val = obj
                    best_sol = cand.copy()

    return best_sol, best_val


def _greedy(demands, capacities, assign_cost):
    """Fallback greedy: largest demand first, cheapest feasible facility."""
    n = len(demands)
    m = len(capacities)
    result = np.zeros(n, dtype=int)
    load = np.zeros(m)
    for i in np.argsort(-demands):
        for f in np.argsort(assign_cost[i]):
            if load[f] + demands[i] <= capacities[f] + 1e-9:
                load[f] += demands[i]
                result[i] = f
                break
    return result


def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    n = len(demands)
    m = len(capacities)

    # Try Lagrangian subgradient optimization
    rng = np.random.default_rng()
    sol, val = _subgradient(demands, capacities, open_cost, assign_cost,
                            n_restarts=3, max_iter=400, rng=rng)

    # Fallback to greedy if Lagrangian failed
    if sol is None:
        sol = _greedy(demands, capacities, assign_cost)

    # Final validation: ensure integrality and capacity feasibility
    sol = np.asarray(sol, dtype=int)
    load = np.bincount(sol, weights=demands, minlength=m)
    if not np.all(load <= capacities + 1e-9):
        # Repair via greedy
        sol = _greedy(demands, capacities, assign_cost)

    return sol
# EVOLVE-BLOCK-END
