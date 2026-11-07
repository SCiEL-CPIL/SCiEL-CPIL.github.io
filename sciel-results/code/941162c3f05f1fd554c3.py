"""Capacitated facility location: Lagrangian relaxation + repair + local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign customers to facilities using:
    1. Lagrangian relaxation on capacity constraints with subgradient updates
    2. Greedy repair for feasibility
    3. Local search (moves and swaps) to reduce cost
    """
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    n_customers = len(demands)
    n_facilities = len(capacities)
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    def total_cost(assignment):
        """Compute total cost: open cost + assignment cost."""
        used = np.zeros(n_facilities, dtype=bool)
        for f in assignment:
            used[f] = True
        oc = open_cost[used].sum()
        ac = assign_cost[np.arange(n_customers), assignment].sum()
        return oc + ac

    def is_feasible(assignment):
        load = np.zeros(n_facilities, dtype=np.float64)
        for i in range(n_customers):
            load[assignment[i]] += demands[i]
        return np.all(load <= capacities + 1e-9)

    # --- Step 1: Lagrangian relaxation ---
    # Dual variables for capacity constraints
    lambda_ = np.zeros(n_facilities, dtype=np.float64)
    best_assignment = None
    best_cost = np.inf
    best_feasible = None
    best_feasible_cost = np.inf

    # Initial step size
    step_size = 1.0
    max_lag_iters = 500

    for it in range(max_lag_iters):
        if time.time() > DEADLINE - 30:
            break

        # Relaxed cost: assign_cost[i,f] + lambda_[f] * demands[i]
        # If facility f is "opened" (has any customer), add open_cost[f]
        # For Lagrangian, we treat opening as part of the relaxed problem.
        # Simpler: just use adjusted cost = assign_cost + lambda * demands
        adjusted = assign_cost + np.outer(demands, lambda_)

        # For each customer, pick the facility with minimum adjusted cost
        assignment = np.argmin(adjusted, axis=1).astype(int)

        # Compute load and check feasibility
        load = np.zeros(n_facilities, dtype=np.float64)
        for i in range(n_customers):
            load[assignment[i]] += demands[i]

        # Compute actual cost (with open cost for used facilities)
        used = np.zeros(n_facilities, dtype=bool)
        for f in assignment:
            used[f] = True
        actual_cost = open_cost[used].sum() + assign_cost[np.arange(n_customers), assignment].sum()

        # Track best feasible
        if np.all(load <= capacities + 1e-9):
            if actual_cost < best_feasible_cost:
                best_feasible_cost = actual_cost
                best_feasible = assignment.copy()

        # Track best overall
        if actual_cost < best_cost:
            best_cost = actual_cost
            best_assignment = assignment.copy()

        # Subgradient update: g_f = load_f - capacity_f
        grad = load - capacities
        # Project lambda to non-negative
        lambda_ = np.maximum(0, lambda_ + step_size * grad)

        # Decrease step size
        step_size *= 0.95

        # Early stop if step size is tiny
        if step_size < 1e-6:
            break

    # --- Step 2: Greedy repair if no feasible solution found ---
    if best_feasible is None:
        # Start from best_assignment and repair
        assignment = best_assignment.copy() if best_assignment is not None else np.zeros(n_customers, dtype=int)
        # Rebuild: sort by demand descending, assign greedily
        order = np.argsort(-demands)
        load = np.zeros(n_facilities, dtype=np.float64)
        assignment = np.zeros(n_customers, dtype=int)
        for i in order:
            placed = False
            # Try cheapest feasible facility
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    load[f] += demands[i]
                    assignment[i] = f
                    placed = True
                    break
            if not placed:
                # Fallback: assign to facility with most remaining capacity
                remaining = capacities - load
                f = int(np.argmax(remaining))
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    load[f] += demands[i]
                    assignment[i] = f
                else:
                    # Should not happen if total capacity >= total demand
                    assignment[i] = f
                    load[f] += demands[i]

        if is_feasible(assignment):
            best_feasible = assignment.copy()
            best_feasible_cost = total_cost(assignment)

    # Use the best feasible assignment
    if best_feasible is not None:
        result = best_feasible.copy()
    else:
        # Last resort: pure greedy
        order = np.argsort(-demands)
        load = np.zeros(n_facilities, dtype=np.float64)
        result = np.zeros(n_customers, dtype=int)
        for i in order:
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    load[f] += demands[i]
                    result[i] = f
                    break
            else:
                f = int(np.argmax(capacities - load))
                load[f] += demands[i]
                result[i] = f

    # --- Step 3: Local search ---
    def local_search(assignment, max_time=120.0):
        """Improve assignment via moves and swaps."""
        assignment = assignment.copy().astype(int)
        load = np.zeros(n_facilities, dtype=np.float64)
        for i in range(n_customers):
            load[assignment[i]] += demands[i]

        used = np.zeros(n_facilities, dtype=bool)
        for f in assignment:
            used[f] = True
        current_cost = open_cost[used].sum() + assign_cost[np.arange(n_customers), assignment].sum()

        improved = True
        while improved and time.time() < max_time:
            improved = False
            # Try moves
            for i in range(n_customers):
                if time.time() > max_time:
                    break
                old_f = assignment[i]
                # Try moving to each other facility
                for new_f in range(n_facilities):
                    if new_f == old_f:
                        continue
                    if load[new_f] + demands[i] > capacities[new_f] + 1e-9:
                        continue

                    # Compute delta cost
                    # Remove i from old_f, add to new_f
                    # Check if old_f becomes empty (saves open_cost)
                    old_load_after = load[old_f] - demands[i]
                    new_load_before = load[new_f]

                    # Open cost change
                    oc_delta = 0.0
                    if old_load_after <= 1e-9 and used[old_f]:
                        oc_delta -= open_cost[old_f]  # facility closes
                    if new_load_before <= 1e-9 and not used[new_f]:
                        oc_delta += open_cost[new_f]  # facility opens

                    # Assignment cost change
                    ac_delta = assign_cost[i, new_f] - assign_cost[i, old_f]

                    delta = oc_delta + ac_delta
                    if delta < -1e-9:
                        # Perform move
                        load[old_f] = old_load_after
                        load[new_f] = new_load_before + demands[i]
                        if old_load_after <= 1e-9:
                            used[old_f] = False
                        if new_load_before <= 1e-9:
                            used[new_f] = True
                        assignment[i] = new_f
                        current_cost += delta
                        improved = True
                        break  # restart scan for this customer

            # Try swaps
            if time.time() > max_time:
                break
            for i in range(n_customers):
                if time.time() > max_time:
                    break
                for j in range(i + 1, n_customers):
                    if time.time() > max_time:
                        break
                    fi = assignment[i]
                    fj = assignment[j]
                    if fi == fj:
                        continue

                    # Check capacity: i goes to fj, j goes to fi
                    # Load at fi after: load[fi] - demands[i] + demands[j]
                    # Load at fj after: load[fj] - demands[j] + demands[i]
                    new_load_fi = load[fi] - demands[i] + demands[j]
                    new_load_fj = load[fj] - demands[j] + demands[i]

                    if new_load_fi > capacities[fi] + 1e-9:
                        continue
                    if new_load_fj > capacities[fj] + 1e-9:
                        continue

                    # Open cost change
                    oc_delta = 0.0
                    if load[fi] - demands[i] <= 1e-9 and used[fi]:
                        # fi might close if j doesn't go there... but j does go to fi
                        # Actually j goes to fi, so fi stays open
                        pass
                    if load[fj] - demands[j] <= 1e-9 and used[fj]:
                        # fj might close if i doesn't go there... but i does go to fj
                        pass

                    # Since both facilities remain open (each gets one customer),
                    # open cost doesn't change unless a facility becomes empty.
                    # After swap: fi has load[fi] - demands[i] + demands[j]
                    #             fj has load[fj] - demands[j] + demands[i]
                    # Neither becomes empty since we're just swapping.
                    # Actually they could become empty if the other demand is 0, but demands >= 5.
                    # So open cost unchanged.

                    ac_delta = (assign_cost[i, fj] + assign_cost[j, fi]) - (assign_cost[i, fi] + assign_cost[j, fj])

                    delta = oc_delta + ac_delta
                    if delta < -1e-9:
                        # Perform swap
                        load[fi] = new_load_fi
                        load[fj] = new_load_fj
                        assignment[i] = fj
                        assignment[j] = fi
                        current_cost += delta
                        improved = True
                        break  # restart

        return assignment, current_cost

    remaining_time = DEADLINE - time.time()
    if remaining_time > 5:
        result, _ = local_search(result, max_time=min(remaining_time, 120.0))

    return result.astype(int)
# EVOLVE-BLOCK-END
