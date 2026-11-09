"""Rewritten program using min-cost flow / successive shortest path with potentials."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
from collections import defaultdict

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign customers to facilities minimizing total cost (open + assignment).
    Uses a min-cost flow formulation with successive shortest path.
    """
    start_time = time.time()
    DEADLINE = 115.0  # seconds, safety margin before 120s limit

    n_customers = len(demands)
    n_facilities = len(capacities)

    # Build a min-cost flow network:
    # Source -> Customer nodes (capacity = demand[i], cost = 0)
    # Customer i -> Facility j (capacity = demand[i], cost = assign_cost[i,j])
    # Facility j -> Sink (capacity = capacities[j], cost = 0)
    # But we also need to handle open_cost: if a facility gets any flow, we pay open_cost[j].
    #
    # Standard trick: split facility node into j_in and j_out.
    # j_in -> j_out has capacity = capacities[j], cost = open_cost[j] for the first unit,
    # and then additional units have cost 0. But flow is integral and we send demand[i] units.
    # Actually, we can model it as: facility j has two edges to sink:
    #   Edge 1: capacity 1, cost = open_cost[j]
    #   Edge 2: capacity = capacities[j] - 1, cost = 0
    # But this only works if we send 1 unit per customer. Since each customer sends demand[i] units,
    # this doesn't directly work.
    #
    # Alternative: Since open_cost is fixed per facility (not per unit), we can use a different approach.
    # Let's use a heuristic + local search approach instead, which is more practical.
    #
    # Actually, let me reconsider. The problem size is small: 100 customers, 20 facilities.
    # We can use a greedy initialization + local search with facility open/close moves.

    # Greedy initialization: sort customers by demand descending, assign to cheapest feasible facility
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    order = np.argsort(-demands)
    for i in order:
        # Try facilities in order of assign_cost
        for f in np.argsort(assign_cost[i]):
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
        else:
            # Fallback: find any facility with remaining capacity
            for f in range(n_facilities):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break

    def total_cost(assignment):
        """Compute total cost: open costs + assignment costs."""
        used = set(assignment)
        open_total = sum(open_cost[f] for f in used)
        assign_total = sum(assign_cost[i, assignment[i]] for i in range(n_customers))
        return open_total + assign_total

    def is_valid(assignment):
        """Check if assignment respects capacities."""
        loads = np.zeros(n_facilities)
        for i in range(n_customers):
            loads[assignment[i]] += demands[i]
        return np.all(loads <= capacities)

    # Local search: try moving individual customers to other facilities
    best_assignment = result.copy()
    best_cost = total_cost(best_assignment)

    # Precompute for efficiency
    best_load = np.zeros(n_facilities)
    for i in range(n_customers):
        best_load[best_assignment[i]] += demands[i]

    improved = True
    while improved and (time.time() - start_time) < DEADLINE:
        improved = False
        # Try moving each customer to a different facility
        for i in range(n_customers):
            if (time.time() - start_time) > DEADLINE:
                break
            current_f = best_assignment[i]
            # Try all other facilities
            for new_f in range(n_facilities):
                if new_f == current_f:
                    continue
                if best_load[new_f] + demands[i] > capacities[new_f]:
                    continue
                # Compute delta cost
                # Assignment cost change
                delta_assign = assign_cost[i, new_f] - assign_cost[i, current_f]
                # Open cost change
                delta_open = 0
                if best_load[current_f] == demands[i]:
                    # Current facility becomes empty
                    delta_open -= open_cost[current_f]
                if best_load[new_f] == 0:
                    # New facility becomes used
                    delta_open += open_cost[new_f]
                delta = delta_assign + delta_open
                if delta < -1e-9:
                    # Apply move
                    best_load[current_f] -= demands[i]
                    best_load[new_f] += demands[i]
                    best_assignment[i] = new_f
                    best_cost += delta
                    improved = True
                    break  # Move to next customer

    # Try 2-opt style: swap two customers' facilities
    if (time.time() - start_time) < DEADLINE:
        improved = True
        while improved and (time.time() - start_time) < DEADLINE:
            improved = False
            for i in range(n_customers):
                if (time.time() - start_time) > DEADLINE:
                    break
                for j in range(i + 1, n_customers):
                    if (time.time() - start_time) > DEADLINE:
                        break
                    fi = best_assignment[i]
                    fj = best_assignment[j]
                    if fi == fj:
                        continue
                    # Check if swapping is feasible
                    if best_load[fi] - demands[i] + demands[j] > capacities[fi]:
                        continue
                    if best_load[fj] - demands[j] + demands[i] > capacities[fj]:
                        continue
                    # Compute delta
                    delta_assign = (assign_cost[i, fj] + assign_cost[j, fi]) - (assign_cost[i, fi] + assign_cost[j, fj])
                    delta_open = 0
                    # Check if any facility becomes empty or newly used
                    if best_load[fi] == demands[i]:
                        # fi becomes empty if we move i away and j doesn't come to fi
                        # But j comes to fi, so fi is still used if demands[j] > 0 (always true)
                        pass
                    if best_load[fj] == demands[j]:
                        pass
                    # More carefully:
                    new_load_fi = best_load[fi] - demands[i] + demands[j]
                    new_load_fj = best_load[fj] - demands[j] + demands[i]
                    if best_load[fi] > 0 and new_load_fi == 0:
                        delta_open -= open_cost[fi]
                    if best_load[fj] > 0 and new_load_fj == 0:
                        delta_open -= open_cost[fj]
                    if best_load[fi] == 0 and new_load_fi > 0:
                        delta_open += open_cost[fi]
                    if best_load[fj] == 0 and new_load_fj > 0:
                        delta_open += open_cost[fj]
                    delta = delta_assign + delta_open
                    if delta < -1e-9:
                        # Apply swap
                        best_load[fi] = new_load_fi
                        best_load[fj] = new_load_fj
                        best_assignment[i] = fj
                        best_assignment[j] = fi
                        best_cost += delta
                        improved = True
                        break
                if improved:
                    break

    # Validate result
    if not is_valid(best_assignment):
        # Fallback to greedy result
        return result

    return best_assignment
# EVOLVE-BLOCK-END
