"""Stronger solver for capacitated facility location with SA."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility.
    Strategy: 
    1. Diverse greedy initializations.
    2. Fast vectorized local search (single moves + pairwise swaps).
    3. Simulated Annealing to escape local minima.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    n_customers = len(demands)
    n_facilities = len(capacities)

    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    def compute_cost(assignment):
        """Compute total cost. Returns inf if infeasible."""
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

    def greedy_assignment(order, rng=None):
        """Greedy: assign in given order to cheapest feasible facility."""
        result = np.full(n_customers, -1, dtype=int)
        load = np.zeros(n_facilities)
        for c in order:
            costs = assign_cost[c].copy()
            feasible = load + demands[c] <= capacities
            costs[~feasible] = np.inf
            if rng is not None:
                feasible_mask = feasible
                if feasible_mask.any():
                    pert = rng.uniform(0, 0.1 * np.mean(costs[feasible_mask]), size=n_facilities)
                    pert[~feasible_mask] = 0
                    costs += pert
            f = int(np.argmin(costs))
            if f >= 0 and feasible[f]:
                result[c] = f
                load[f] += demands[c]
            else:
                return None
        return result

    def local_search(assignment, max_passes=100):
        """
        Fast local search: single reassignments + pairwise swaps.
        """
        assignment = assignment.copy()
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            load[int(assignment[c])] += demands[c]

        current_cost = compute_cost(assignment)
        if current_cost == np.inf:
            return assignment

        best_cost = current_cost
        best_assignment = assignment.copy()

        for _pass in range(max_passes):
            if time.time() > deadline - 5:
                break

            improved = False

            # --- Single reassignment pass ---
            for c in range(n_customers):
                if time.time() > deadline - 5:
                    break
                old_f = int(assignment[c])
                load[old_f] -= demands[c]

                # Vectorized: compute delta for all facilities
                deltas = assign_cost[c] - assign_cost[c, old_f]
                newly_used = (load == 0)
                deltas[newly_used] += open_cost[newly_used]
                if load[old_f] == 0:
                    deltas -= open_cost[old_f]

                feasible = load + demands[c] <= capacities + 1e-9
                feasible[old_f] = False
                deltas[~feasible] = np.inf

                best_idx = int(np.argmin(deltas))
                best_delta = deltas[best_idx]

                if best_delta < -1e-9 and feasible[best_idx]:
                    assignment[c] = best_idx
                    load[best_idx] += demands[c]
                    current_cost += best_delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_assignment = assignment.copy()
                else:
                    load[old_f] += demands[c]

            # --- Pairwise swap pass ---
            if time.time() > deadline - 5:
                break

            # Full pass of swaps
            for c1 in range(n_customers):
                if time.time() > deadline - 5:
                    break
                f1 = int(assignment[c1])
                for c2 in range(c1 + 1, n_customers):
                    if time.time() > deadline - 5:
                        break
                    f2 = int(assignment[c2])
                    if f1 == f2:
                        continue

                    # Check feasibility
                    load_f1_new = load[f1] - demands[c1] + demands[c2]
                    load_f2_new = load[f2] - demands[c2] + demands[c1]
                    if load_f1_new > capacities[f1] + 1e-9 or load_f2_new > capacities[f2] + 1e-9:
                        continue

                    # Cost delta
                    delta = (assign_cost[c1, f2] + assign_cost[c2, f1]) - \
                            (assign_cost[c1, f1] + assign_cost[c2, f2])

                    # Open cost changes
                    if load[f1] > 0 and load_f1_new == 0:
                        delta -= open_cost[f1]
                    elif load[f1] == 0 and load_f1_new > 0:
                        delta += open_cost[f1]
                    
                    if load[f2] > 0 and load_f2_new == 0:
                        delta -= open_cost[f2]
                    elif load[f2] == 0 and load_f2_new > 0:
                        delta += open_cost[f2]

                    if delta < -1e-9:
                        assignment[c1] = f2
                        assignment[c2] = f1
                        load[f1] = load_f1_new
                        load[f2] = load_f2_new
                        current_cost += delta
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_assignment = assignment.copy()

            if not improved:
                break

        return best_assignment

    def simulated_annealing(assignment, t_start=100.0, t_end=0.1, max_iters=50000):
        """
        Simulated Annealing to escape local minima.
        """
        assignment = assignment.copy()
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            load[int(assignment[c])] += demands[c]
        
        current_cost = compute_cost(assignment)
        if current_cost == np.inf:
            return assignment

        best_assignment = assignment.copy()
        best_cost = current_cost

        rng = np.random.RandomState(123)
        
        # Estimate step size for temperature
        # Average assignment cost is around 100 * sqrt(0.5) ~ 70.
        # Open cost is 50-150.
        # T_start should be high enough to accept some worse moves.
        
        for i in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            # Cooling schedule
            T = t_start * (t_end / t_start) ** (i / max(1, max_iters - 1))
            
            # Pick random customer
            c = rng.randint(n_customers)
            old_f = int(assignment[c])
            
            # Find feasible facilities
            load[old_f] -= demands[c]
            feasible = np.where(load + demands[c] <= capacities + 1e-9)[0]
            
            if len(feasible) == 0:
                load[old_f] += demands[c]
                continue
            
            # Pick random feasible facility
            new_f = int(feasible[rng.randint(len(feasible))])
            
            if new_f == old_f:
                load[old_f] += demands[c]
                continue
            
            # Compute delta
            delta = assign_cost[c, new_f] - assign_cost[c, old_f]
            
            # Open cost changes
            if load[new_f] == 0:
                delta += open_cost[new_f]
            if load[old_f] == 0:
                delta -= open_cost[old_f]
            
            # Acceptance probability
            if delta < 0:
                accept = True
            else:
                # Avoid overflow in exp
                if delta > T * 100: 
                    accept = False
                else:
                    accept = rng.random() < np.exp(-delta / T)
            
            if accept:
                assignment[c] = new_f
                load[new_f] += demands[c]
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_assignment = assignment.copy()
            else:
                load[old_f] += demands[c]
                
        return best_assignment

    # Phase 1: Generate diverse initial solutions
    best_solution = None
    best_cost = np.inf

    orderings = [
        np.arange(n_customers),
        np.argsort(-demands),
        np.argsort(demands),
        np.argsort(assign_cost.sum(axis=1)),
        np.argsort(assign_cost.sum(axis=1) / demands),
    ]

    for order in orderings:
        if time.time() > deadline - 60:
            break
        sol = greedy_assignment(order)
        if sol is not None:
            cost = compute_cost(sol)
            if cost < best_cost:
                best_cost = cost
                best_solution = sol.copy()

    # Randomized greedy
    rng = np.random.RandomState(42)
    n_random = 30
    for i in range(n_random):
        if time.time() > deadline - 60:
            break
        order = rng.permutation(n_customers)
        sol = greedy_assignment(order, rng)
        if sol is not None:
            cost = compute_cost(sol)
            if cost < best_cost:
                best_cost = cost
                best_solution = sol.copy()

    # Phase 2: Local search on best
    if best_solution is not None and time.time() < deadline - 30:
        improved = local_search(best_solution, max_passes=50)
        cost = compute_cost(improved)
        if cost < best_cost:
            best_cost = cost
            best_solution = improved

    # Phase 3: Simulated Annealing on the best local solution
    if best_solution is not None and time.time() < deadline - 5:
        sa_sol = simulated_annealing(best_solution, t_start=50.0, t_end=0.5, max_iters=20000)
        cost = compute_cost(sa_sol)
        if cost < best_cost:
            best_cost = cost
            best_solution = sa_sol
        # Run local search again on SA result to polish
        if time.time() < deadline - 2:
            final_sol = local_search(best_solution, max_passes=20)
            cost = compute_cost(final_sol)
            if cost < best_cost:
                best_cost = cost
                best_solution = final_sol

    # Final check
    if best_solution is None:
        best_solution = greedy_assignment(np.argsort(-demands))
        if best_solution is None:
            best_solution = np.zeros(n_customers, dtype=int)

    return best_solution.astype(int)
# EVOLVE-BLOCK-END
