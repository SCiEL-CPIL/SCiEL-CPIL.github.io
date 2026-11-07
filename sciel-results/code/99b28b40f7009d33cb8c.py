"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    N = n_customers
    F = n_facilities
    
    def greedy_fallback():
        result = np.zeros(N, dtype=int)
        load = np.zeros(F)
        for i in np.argsort(-demands):
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break
        return result, load
    
    # Lagrangian relaxation approach
    # Multipliers for capacity constraints
    lam = np.zeros(F)
    
    best_assignment = None
    best_cost = np.inf
    
    def solve_subproblem(lam):
        """Solve the Lagrangian subproblem for given multipliers."""
        # Effective cost for customer i, facility j: assign_cost[i,j] + lam[j] * demands[i]
        eff_cost = assign_cost + np.outer(demands, lam)
        
        # Independent assignment
        assignment = np.argmin(eff_cost, axis=1)
        
        # Compute load
        load = np.zeros(F)
        for i in range(N):
            load[assignment[i]] += demands[i]
        
        # Total cost including open costs
        used = np.unique(assignment)
        total_cost = np.sum(eff_cost[np.arange(N), assignment]) + np.sum(open_cost[used])
        
        # Try to improve: for facilities with high open cost and low load, 
        # reassign their customers to next-best facilities
        # Simple improvement: iterate a few times
        for _ in range(3):
            improved = False
            for j in used.copy():
                if load[j] == 0:
                    continue
                # Customers at facility j
                custs_j = np.where(assignment == j)[0]
                if len(custs_j) == 0:
                    continue
                # Check if closing j and reassigning is beneficial
                # For each customer at j, find best alternative
                alt_cost = 0.0
                feasible = True
                new_load = load.copy()
                new_load[j] = 0
                for i in custs_j:
                    # Find best facility k != j with capacity
                    best_k = -1
                    best_c = np.inf
                    for k in range(F):
                        if k == j:
                            continue
                        if new_load[k] + demands[i] <= capacities[k]:
                            c = eff_cost[i, k]
                            if c < best_c:
                                best_c = c
                                best_k = k
                    if best_k == -1:
                        feasible = False
                        break
                    alt_cost += best_c
                    new_load[best_k] += demands[i]
                
                if feasible:
                    # Compare: current cost contribution of j's customers + open_cost[j]
                    # vs alt_cost (no open_cost[j] if j becomes empty)
                    current_contrib = np.sum(eff_cost[custs_j, j]) + open_cost[j]
                    if alt_cost < current_contrib - 1e-9:
                        # Reassign
                        new_assignment = assignment.copy()
                        new_load = load.copy()
                        new_load[j] = 0
                        for i in custs_j:
                            best_k = -1
                            best_c = np.inf
                            for k in range(F):
                                if k == j:
                                    continue
                                if new_load[k] + demands[i] <= capacities[k]:
                                    c = eff_cost[i, k]
                                    if c < best_c:
                                        best_c = c
                                        best_k = k
                            if best_k != -1:
                                new_assignment[i] = best_k
                                new_load[best_k] += demands[i]
                        assignment = new_assignment
                        load = new_load
                        improved = True
            
            if not improved:
                break
        
        # Recompute load and cost
        load = np.zeros(F)
        for i in range(N):
            load[assignment[i]] += demands[i]
        
        used = np.unique(assignment)
        total_cost = np.sum(eff_cost[np.arange(N), assignment]) + np.sum(open_cost[used])
        
        return assignment, load, total_cost
    
    # Subgradient method
    step_size = 1.0
    best_lam = lam.copy()
    
    for iteration in range(500):
        if time.time() > deadline - 30.0:
            break
        
        assignment, load, total_cost = solve_subproblem(lam)
        
        # Check if feasible
        if np.all(load <= capacities + 1e-6):
            # Feasible! Compute actual cost
            used = np.unique(assignment)
            actual_cost = np.sum(assign_cost[np.arange(N), assignment]) + np.sum(open_cost[used])
            if actual_cost < best_cost:
                best_cost = actual_cost
                best_assignment = assignment.copy()
        
        # Subgradient
        subgrad = load - capacities
        
        # Update multipliers
        lam_new = np.maximum(0, lam + step_size * subgrad)
        
        # Step size update (Polyak-style if we have upper bound)
        if best_assignment is not None:
            # Use a simple decreasing step
            step_size = max(0.1, 1.0 / (1.0 + 0.01 * iteration))
        else:
            step_size = max(0.01, 1.0 / (1.0 + 0.05 * iteration))
        
        lam = lam_new
        
        # Early stop if no improvement in many iterations
        if iteration > 50 and np.allclose(lam, best_lam, atol=1e-6):
            break
    
    # Get a feasible solution
    if best_assignment is None:
        # Use greedy fallback
        best_assignment, _ = greedy_fallback()
    
    # Local search to polish
    def calc_cost(assignment):
        total_assign = np.sum(assign_cost[np.arange(N), assignment])
        used = np.unique(assignment)
        total_open = np.sum(open_cost[used])
        return total_assign + total_open
    
    def is_valid(assignment):
        for f in range(F):
            mask = assignment == f
            if mask.any():
                if demands[mask].sum() > capacities[f] + 1e-6:
                    return False
        return True
    
    if not is_valid(best_assignment):
        # Force feasibility with greedy
        best_assignment, _ = greedy_fallback()
    
    best_cost = calc_cost(best_assignment)
    load = np.zeros(F)
    for i in range(N):
        load[best_assignment[i]] += demands[i]
    
    # Local search: moves and swaps
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # Moves
        for i in range(N):
            if time.time() > deadline - 5.0:
                break
            current_f = best_assignment[i]
            for f in range(F):
                if f == current_f:
                    continue
                if load[f] + demands[i] <= capacities[f]:
                    new_cost = best_cost - assign_cost[i, current_f] + assign_cost[i, f]
                    if np.sum(best_assignment == current_f) == 1:
                        new_cost -= open_cost[current_f]
                    if np.sum(best_assignment == f) == 0:
                        new_cost += open_cost[f]
                    if new_cost < best_cost - 1e-9:
                        load[current_f] -= demands[i]
                        load[f] += demands[i]
                        best_assignment[i] = f
                        best_cost = new_cost
                        improved = True
                        break
        
        # Swaps
        if time.time() > deadline - 5.0:
            break
        for i in range(N):
            if time.time() > deadline - 5.0:
                break
            for j in range(i + 1, N):
                if time.time() > deadline - 5.0:
                    break
                fi = best_assignment[i]
                fj = best_assignment[j]
                if fi == fj:
                    continue
                if (load[fi] - demands[i] + demands[j] <= capacities[fi] and
                    load[fj] - demands[j] + demands[i] <= capacities[fj]):
                    new_cost = (best_cost - assign_cost[i, fi] - assign_cost[j, fj] +
                                assign_cost[i, fj] + assign_cost[j, fi])
                    if new_cost < best_cost - 1e-9:
                        load[fi] = load[fi] - demands[i] + demands[j]
                        load[fj] = load[fj] - demands[j] + demands[i]
                        best_assignment[i] = fj
                        best_assignment[j] = fi
                        best_cost = new_cost
                        improved = True
    
    # Final verification
    if not is_valid(best_assignment):
        # Last resort: greedy
        fallback, _ = greedy_fallback()
        return fallback
    
    return best_assignment
# EVOLVE-BLOCK-END
