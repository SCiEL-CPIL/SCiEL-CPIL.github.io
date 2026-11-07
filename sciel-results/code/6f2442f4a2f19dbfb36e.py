"""Capacitated facility location: strong heuristic with multi-start and local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility respecting capacity constraints.
    Uses multi-start greedy construction with local search (relocate + swap).
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Deadline guard: 145s safety margin out of 150s
    deadline = time.time() + 145.0
    
    # Convert to numpy arrays for speed
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    def compute_cost(assignment, load, fac_counts):
        """Compute total cost: open costs + assignment costs."""
        total_assign = np.sum(assign_cost[np.arange(n_customers), assignment])
        used_facilities = fac_counts > 0
        total_open = np.sum(open_cost[used_facilities])
        return total_assign + total_open
    
    def greedy_init(seed_order=None):
        """Greedy construction: assign customers to cheapest feasible facility."""
        assignment = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities, dtype=np.float64)
        
        if seed_order is None:
            # Sort customers by demand descending
            order = np.argsort(-demands)
        else:
            order = seed_order
        
        for i in order:
            # Get facilities sorted by assignment cost for this customer
            fac_order = np.argsort(assign_cost[i])
            assigned = False
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    assignment[i] = f
                    assigned = True
                    break
            if not assigned:
                # Fallback: assign to facility with most remaining capacity
                remaining = capacities - load
                f = np.argmax(remaining)
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    assignment[i] = f
                else:
                    # Force assign (should not happen in valid instances)
                    assignment[i] = f
                    load[f] += demands[i]
        
        # Compute facility counts
        fac_counts = np.zeros(n_facilities, dtype=int)
        for i in range(n_customers):
            fac_counts[assignment[i]] += 1
        
        return assignment, load, fac_counts
    
    def local_search(assignment, load, fac_counts, max_time):
        """Perform local search with relocate and 2-swap moves."""
        current_cost = compute_cost(assignment, load, fac_counts)
        best_cost = current_cost
        best_assignment = assignment.copy()
        best_load = load.copy()
        best_fac_counts = fac_counts.copy()
        
        deadline_ls = time.time() + max_time
        
        # Relocate moves
        no_improve = 0
        while time.time() < deadline_ls and no_improve < 100:
            no_improve += 1
            # Try relocating each customer
            for i in range(n_customers):
                if time.time() > deadline_ls:
                    break
                cf = assignment[i]
                # Try all other facilities sorted by assignment cost
                for f in np.argsort(assign_cost[i]):
                    if f == cf:
                        continue
                    if load[f] + demands[i] > capacities[f]:
                        continue
                    
                    # Delta assignment cost
                    d_assign = assign_cost[i, f] - assign_cost[i, cf]
                    
                    # Delta open cost
                    d_open = 0.0
                    if fac_counts[cf] == 1:
                        d_open -= open_cost[cf]  # We're closing cf
                    if fac_counts[f] == 0:
                        d_open += open_cost[f]  # We're opening f
                    
                    if d_assign + d_open < -1e-9:
                        # Make the move
                        assignment[i] = f
                        load[cf] -= demands[i]
                        load[f] += demands[i]
                        fac_counts[cf] -= 1
                        fac_counts[f] += 1
                        current_cost += d_assign + d_open
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_assignment = assignment.copy()
                            best_load = load.copy()
                            best_fac_counts = fac_counts.copy()
                        no_improve = 0
                        break  # Move to next customer
        
        # 2-swap moves: swap two customers between facilities
        if time.time() < deadline_ls:
            swapped = True
            while swapped and time.time() < deadline_ls:
                swapped = False
                for i in range(n_customers):
                    if time.time() > deadline_ls:
                        break
                    for j in range(i + 1, n_customers):
                        if time.time() > deadline_ls:
                            break
                        fi = assignment[i]
                        fj = assignment[j]
                        if fi == fj:
                            continue
                        
                        # Check capacity for swap
                        if load[fi] - demands[i] + demands[j] > capacities[fi]:
                            continue
                        if load[fj] - demands[j] + demands[i] > capacities[fj]:
                            continue
                        
                        # Delta assignment cost
                        d_assign = (assign_cost[i, fj] + assign_cost[j, fi]) - \
                                   (assign_cost[i, fi] + assign_cost[j, fj])
                        
                        # For 2-swaps between two non-empty facilities, open cost doesn't change
                        if d_assign < -1e-9:
                            # Make the swap
                            assignment[i] = fj
                            assignment[j] = fi
                            load[fi] = load[fi] - demands[i] + demands[j]
                            load[fj] = load[fj] - demands[j] + demands[i]
                            current_cost += d_assign
                            if current_cost < best_cost:
                                best_cost = current_cost
                                best_assignment = assignment.copy()
                                best_load = load.copy()
                                best_fac_counts = fac_counts.copy()
                            swapped = True
                            no_improve = 0
        
        return best_assignment, best_cost, best_load, best_fac_counts
    
    # --- Multi-start approach ---
    best_assignment = None
    best_cost = float('inf')
    best_load = None
    best_fac_counts = None
    
    # First: standard greedy (largest demand first)
    assignment, load, fac_counts = greedy_init()
    assignment, cost, load, fac_counts = local_search(assignment, load, fac_counts, 
                                                       min(30.0, deadline - time.time() - 5.0))
    if cost < best_cost:
        best_cost = cost
        best_assignment = assignment
        best_load = load
        best_fac_counts = fac_counts
    
    # Try different random orderings
    rng = np.random.RandomState(42)
    remaining_time = deadline - time.time() - 5.0
    num_starts = min(5, int(remaining_time / 10.0) + 1)
    
    for start_idx in range(num_starts):
        if time.time() > deadline - 5.0:
            break
        
        # Create a random ordering with some bias towards large demands
        order = np.arange(n_customers)
        # Mix of random and demand-sorted
        if start_idx % 2 == 0:
            # Mostly random with some demand bias
            weights = demands ** 0.5
            weights /= weights.sum()
            order = rng.choice(n_customers, size=n_customers, replace=False, p=weights)
        else:
            # Random shuffle
            rng.shuffle(order)
        
        assignment, load, fac_counts = greedy_init(seed_order=order)
        
        # Local search with remaining time
        time_per_search = (deadline - time.time() - 5.0) / (num_starts - start_idx)
        assignment, cost, load, fac_counts = local_search(assignment, load, fac_counts,
                                                           min(time_per_search, 20.0))
        if cost < best_cost:
            best_cost = cost
            best_assignment = assignment
            best_load = load
            best_fac_counts = fac_counts
    
    # Final validation
    if best_assignment is None:
        # Fallback to simple greedy
        assignment, load, fac_counts = greedy_init()
        best_assignment = assignment
    else:
        # Verify capacity constraints
        final_load = np.zeros(n_facilities, dtype=np.float64)
        for i in range(n_customers):
            final_load[best_assignment[i]] += demands[i]
        
        if np.any(final_load > capacities + 1e-9):
            # Redo greedy
            assignment, load, fac_counts = greedy_init()
            best_assignment = assignment
    
    return best_assignment.astype(int)
# EVOLVE-BLOCK-END
