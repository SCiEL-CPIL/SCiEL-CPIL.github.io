"""Stronger solver for capacitated facility location with ILS."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility.
    Strategy: 
    1. Diverse greedy initializations.
    2. Fast vectorized local search (single moves).
    3. Iterated Local Search (ILS) with perturbation.
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
        Fast local search: single reassignments (best improvement).
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

            # --- Single reassignment pass (Best Improvement) ---
            # We iterate through customers. For each, we find the best facility.
            # To be efficient, we check if the current facility is the best.
            # If not, and the delta is negative, we move.
            # We do this in a single pass. If we move, we continue.
            # This is "First Improvement" if we break, "Best Improvement" if we track best.
            # Best Improvement per pass is safer but slightly slower.
            
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

            if not improved:
                break

        return best_assignment

    def perturb(assignment, load, n_moves=5):
        """
        Perturb the solution by moving n_moves customers to random feasible facilities.
        Returns new assignment, load, and cost.
        """
        assignment = assignment.copy()
        load = load.copy()
        cost = compute_cost(assignment)
        
        if cost == np.inf:
            return assignment, load, cost

        rng = np.random.RandomState(42)
        
        for _ in range(n_moves):
            # Pick random customer
            c = rng.randint(n_customers)
            old_f = int(assignment[c])
            
            # Remove from old facility
            load[old_f] -= demands[c]
            
            # Find feasible facilities
            feasible = np.where(load + demands[c] <= capacities + 1e-9)[0]
            
            if len(feasible) == 0:
                # Revert if not feasible (should not happen if we check)
                load[old_f] += demands[c]
                continue
                
            # Pick random feasible facility (not the old one if possible)
            if len(feasible) > 1:
                feasible_mask = feasible != old_f
                if feasible_mask.any():
                    feasible = feasible[feasible_mask]
            
            new_f = int(feasible[rng.randint(len(feasible))])
            
            # Assign to new facility
            assignment[c] = new_f
            load[new_f] += demands[c]
            
            # Update cost
            # Delta = assign_cost[c, new_f] - assign_cost[c, old_f] + open_cost changes
            # It's easier to just recompute or update carefully.
            # Let's just recompute at the end or track carefully.
            # For simplicity in perturbation, we can just recompute cost after all moves.
            
        cost = compute_cost(assignment)
        return assignment, load, cost

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
    n_random = 20
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
        improved = local_search(best_solution, max_passes=100)
        cost = compute_cost(improved)
        if cost < best_cost:
            best_cost = cost
            best_solution = improved

    # Phase 3: Iterated Local Search
    if best_solution is not None and time.time() < deadline - 5:
        current_sol = best_solution.copy()
        current_cost = best_cost
        current_load = np.zeros(n_facilities)
        for c in range(n_customers):
            current_load[int(current_sol[c])] += demands[c]
        
        rng_ils = np.random.RandomState(123)
        
        while time.time() < deadline - 5:
            # Perturb
            n_moves = rng_ils.randint(3, 8)
            perturbed_sol, perturbed_load, perturbed_cost = perturb(current_sol, current_load, n_moves)
            
            if perturbed_cost == np.inf:
                continue
                
            # Local Search on perturbed
            ls_sol = local_search(perturbed_sol, max_passes=50)
            ls_cost = compute_cost(ls_sol)
            
            if ls_cost < current_cost:
                current_sol = ls_sol
                current_cost = ls_cost
                current_load = np.zeros(n_facilities)
                for c in range(n_customers):
                    current_load[int(current_sol[c])] += demands[c]
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_solution = current_sol.copy()
            # Accept worse solutions with some probability to escape local minima?
            # Simple ILS usually only accepts improvements or uses a threshold.
            # Let's try accepting if not too much worse.
            # Or just restart from best if stuck?
            
    # Final check
    if best_solution is None:
        best_solution = greedy_assignment(np.argsort(-demands))
        if best_solution is None:
            best_solution = np.zeros(n_customers, dtype=int)

    return best_solution.astype(int)
# EVOLVE-BLOCK-END
