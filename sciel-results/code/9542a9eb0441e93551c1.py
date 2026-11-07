import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility.
    Uses multi-start local search with vectorized delta calculation.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

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

    def local_search(assignment):
        """
        Improve assignment via single reassignments.
        Uses vectorized calculation of delta costs for each customer.
        """
        assignment = assignment.copy()
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            load[int(assignment[c])] += demands[c]
        
        used = np.zeros(n_facilities, dtype=bool)
        for f in range(n_facilities):
            if load[f] > 0:
                used[f] = True
        
        current_cost = 0.0
        for c in range(n_customers):
            f = int(assignment[c])
            current_cost += assign_cost[c, f]
        for f in range(n_facilities):
            if used[f]:
                current_cost += open_cost[f]

        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            improved = False
            # Iterate customers in random order to break symmetry
            # Fixed order is also fine, but random might help escape cycles (though 1-swap doesn't cycle if strictly decreasing)
            for c in range(n_customers):
                old_f = int(assignment[c])
                d_c = demands[c]
                
                # Calculate delta for moving to all facilities
                # Base delta: assign_cost[c, f] - assign_cost[c, old_f]
                deltas = assign_cost[c] - assign_cost[c, old_f]
                
                # Opening cost adjustments
                # If moving to a new facility f that is currently empty, we pay open_cost[f]
                # If leaving old_f and it becomes empty, we save open_cost[old_f]
                
                # Mask infeasible facilities
                feasible_mask = load + d_c <= capacities
                feasible_mask[old_f] = False # Don't move to self
                
                if not np.any(feasible_mask):
                    continue

                # Adjust deltas for opening costs
                # This is tricky to vectorize perfectly without conditionals per facility, 
                # but we can approximate or compute exactly.
                
                # Exact calculation:
                # delta(f) = assign_cost[c, f] - assign_cost[c, old_f] 
                #            + (open_cost[f] if load[f]==0 else 0)
                #            - (open_cost[old_f] if load[old_f]-d_c==0 else 0)
                
                # Let's compute this vectorized
                # new_open_cost = open_cost * (load == 0)
                # old_save_cost = open_cost[old_f] if (load[old_f] - d_c == 0) else 0
                
                # Vectorized new open cost impact
                # If load[f] is 0, adding d_c makes it non-zero. We pay open_cost[f].
                # If load[f] is > 0, we pay 0 extra.
                new_open_impact = np.where(load == 0, open_cost, 0.0)
                
                old_save_impact = 0.0
                if load[old_f] - d_c <= 1e-9:
                    old_save_impact = open_cost[old_f]
                
                total_deltas = deltas + new_open_impact - old_save_impact
                
                # Apply feasibility mask
                total_deltas[~feasible_mask] = np.inf
                
                best_f_idx = np.argmin(total_deltas)
                best_delta = total_deltas[best_f_idx]
                
                if best_delta < -1e-9:
                    # Perform move
                    load[old_f] -= d_c
                    if load[old_f] <= 1e-9:
                        used[old_f] = False
                        load[old_f] = 0.0 # Clean up float error
                    
                    load[best_f_idx] += d_c
                    if not used[best_f_idx]:
                        used[best_f_idx] = True
                        load[best_f_idx] = 0.0 # Wait, load was 0, now it's d_c
                        # Actually, if used was False, load was 0.
                        # So load[best_f_idx] becomes d_c.
                    
                    assignment[c] = best_f_idx
                    current_cost += best_delta
                    improved = True
        
        return assignment, current_cost

    def greedy_assignment(order):
        """Greedy: assign in given order to cheapest feasible facility."""
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

    best_assignment = None
    best_cost = np.inf

    # Multiple greedy orderings
    # 1. Natural order
    # 2. Sorted by demand (largest first)
    # 3. Sorted by average assignment cost
    # 4. Random permutations
    
    orderings = [
        np.arange(n_customers),
        np.argsort(-demands),
        np.argsort(assign_cost.mean(axis=1)),
    ]
    
    # Add a few random orderings
    rng = np.random.RandomState(42)
    for _ in range(10):
        orderings.append(rng.permutation(n_customers))

    for order in orderings:
        if time.time() > deadline - 10:
            break
        sol = greedy_assignment(order)
        if sol is not None:
            sol, cost = local_search(sol)
            if cost < best_cost:
                best_cost = cost
                best_assignment = sol.copy()

    if best_assignment is None:
        # Last resort fallback: assign to facility with most remaining capacity
        best_assignment = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            f = int(np.argmax(capacities - load))
            best_assignment[c] = f
            load[f] += demands[c]

    return best_assignment.astype(int)
