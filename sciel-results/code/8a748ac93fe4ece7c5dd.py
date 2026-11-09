"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    n_customers = len(demands)
    n_facilities = len(capacities)

    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    # Greedy initial solution
    result = np.zeros(n_customers, dtype=np.int64)
    load = np.zeros(n_facilities, dtype=np.int64)

    order = np.argsort(-demands)
    for i in order:
        feasible = []
        for f in range(n_facilities):
            if load[f] + demands[i] <= capacities[f]:
                feasible.append(f)

        if not feasible:
            remaining = capacities - load
            best_f = np.argmax(remaining)
            load[best_f] += demands[i]
            result[i] = best_f
        else:
            costs = assign_cost[i, feasible]
            min_idx = np.argmin(costs)
            best_f = feasible[min_idx]
            load[best_f] += demands[i]
            result[i] = best_f

    def compute_total_cost(assignment, loads):
        total = 0.0
        used = set(assignment.tolist())
        for f in used:
            total += open_cost[f]
        for i in range(n_customers):
            total += assign_cost[i, assignment[i]]
        return total

    current_cost = compute_total_cost(result, load)

    # Vectorized local search with first-improvement strategy
    # We iterate until no improvement or time limit

    def try_improve(assignment, loads, cur_cost):
        """Single pass of local search. Returns (new_assignment, new_loads, new_cost, improved)."""
        assignment = assignment.copy()
        loads = loads.copy()
        improved = False
        best_cost = cur_cost

        # First-improvement single moves
        for i in range(n_customers):
            if time.time() > deadline:
                break
            cur_f = assignment[i]
            # Find all facilities that can accept customer i
            feasible_mask = (loads + demands[i]) <= capacities
            feasible_mask[cur_f] = False
            feasible_indices = np.where(feasible_mask)[0]

            if len(feasible_indices) == 0:
                continue

            # Compute cost for each feasible facility
            # Cost change = assign_cost[i, f] - assign_cost[i, cur_f] + open_cost_delta
            base_change = assign_cost[i, feasible_indices] - assign_cost[i, cur_f]

            # Open cost delta:
            # If cur_f becomes empty, we save open_cost[cur_f]
            # If f was empty, we add open_cost[f]
            cur_will_be_empty = (loads[cur_f] == demands[i])
            f_was_empty = (loads[feasible_indices] == 0)

            open_delta = np.zeros(len(feasible_indices))
            if cur_will_be_empty:
                open_delta -= open_cost[cur_f]
            open_delta[f_was_empty] += open_cost[feasible_indices[f_was_empty]]

            total_change = base_change + open_delta

            min_idx = np.argmin(total_change)
            if total_change[min_idx] < -1e-9:
                best_f = feasible_indices[min_idx]
                loads[cur_f] -= demands[i]
                loads[best_f] += demands[i]
                assignment[i] = best_f
                best_cost += total_change[min_idx]
                improved = True

        if time.time() > deadline:
            return assignment, loads, best_cost, improved

        # Second pass: swaps between pairs of customers at different facilities
        # For efficiency, we only consider swaps where the cost change is negative
        # Group customers by facility
        facility_customers = [[] for _ in range(n_facilities)]
        for i in range(n_customers):
            facility_customers[assignment[i]].append(i)

        for f1 in range(n_facilities):
            if time.time() > deadline:
                break
            if not facility_customers[f1]:
                continue
            for f2 in range(f1 + 1, n_facilities):
                if time.time() > deadline:
                    break
                if not facility_customers[f2]:
                    continue

                # Try swaps between customers in f1 and f2
                # For each pair (i in f1, j in f2), check if swap is feasible and improves cost
                # This is O(|f1| * |f2|) per pair of facilities
                # To keep it fast, limit the number of swaps we try per facility pair
                # We'll try a few random pairs and also the best-looking ones

                # Compute cost change for all pairs efficiently
                # For i in f1, j in f2:
                # cost_change = (assign_cost[i,f2] - assign_cost[i,f1]) + (assign_cost[j,f1] - assign_cost[j,f2])
                #              + open_cost_delta

                i_list = facility_customers[f1]
                j_list = facility_customers[f2]

                # If lists are large, subsample
                max_pairs = 50
                if len(i_list) * len(j_list) > max_pairs:
                    # Random subsampling
                    idx_i = np.random.randint(0, len(i_list), min(len(i_list), 10))
                    idx_j = np.random.randint(0, len(j_list), min(len(j_list), 10))
                    i_list = [i_list[k] for k in idx_i]
                    j_list = [j_list[k] for k in idx_j]

                for i in i_list:
                    if time.time() > deadline:
                        break
                    for j in j_list:
                        # Check feasibility
                        new_load_f1 = loads[f1] - demands[i] + demands[j]
                        new_load_f2 = loads[f2] - demands[j] + demands[i]

                        if new_load_f1 > capacities[f1] or new_load_f2 > capacities[f2]:
                            continue

                        cost_change = (assign_cost[i, f2] - assign_cost[i, f1]) + \
                                      (assign_cost[j, f1] - assign_cost[j, f2])

                        # Open cost delta
                        f1_was_empty = (loads[f1] == 0)
                        f2_was_empty = (loads[f2] == 0)
                        f1_will_be_empty = (new_load_f1 == 0)
                        f2_will_be_empty = (new_load_f2 == 0)

                        if f1_will_be_empty and not f1_was_empty:
                            cost_change -= open_cost[f1]
                        elif not f1_will_be_empty and f1_was_empty:
                            cost_change += open_cost[f1]

                        if f2_will_be_empty and not f2_was_empty:
                            cost_change -= open_cost[f2]
                        elif not f2_will_be_empty and f2_was_empty:
                            cost_change += open_cost[f2]

                        if cost_change < -1e-9:
                            # Apply swap
                            loads[f1] = new_load_f1
                            loads[f2] = new_load_f2
                            assignment[i] = f2
                            assignment[j] = f1
                            best_cost += cost_change
                            improved = True
                            # Update facility customers lists
                            # Remove i from f1's list (already moved to f2)
                            # Remove j from f2's list (already moved to f1)
                            facility_customers[f1].remove(i)
                            facility_customers[f2].remove(j)
                            facility_customers[f2].append(i)
                            facility_customers[f1].append(j)

        return assignment, loads, best_cost, improved

    # Run local search
    for iteration in range(100):
        if time.time() > deadline:
            break
        new_assignment, new_loads, new_cost, improved = try_improve(result, load, current_cost)
        if not improved:
            break
        result = new_assignment
        load = new_loads
        current_cost = new_cost

    # Final feasibility check and fix
    final_load = np.zeros(n_facilities, dtype=np.int64)
    for i in range(n_customers):
        f = result[i]
        if f < 0 or f >= n_facilities:
            for f2 in range(n_facilities):
                if load[f2] + demands[i] <= capacities[f2]:
                    result[i] = f2
                    f = f2
                    break
        final_load[f] += demands[i]

    # If any facility is over capacity, try to fix by moving customers
    for f in range(n_facilities):
        if final_load[f] > capacities[f]:
            # Move customers from f to other facilities
            excess = final_load[f] - capacities[f]
            # Find customers at f, sorted by demand ascending
            customers_at_f = [i for i in range(n_customers) if result[i] == f]
            customers_at_f.sort(key=lambda x: demands[x])
            moved = 0
            for i in customers_at_f:
                if moved >= excess:
                    break
                # Find a facility that can accept this customer
                for f2 in range(n_facilities):
                    if f2 == f:
                        continue
                    if final_load[f2] + demands[i] <= capacities[f2]:
                        final_load[f] -= demands[i]
                        final_load[f2] += demands[i]
                        result[i] = f2
                        moved += demands[i]
                        break

    return result
# EVOLVE-BLOCK-END
