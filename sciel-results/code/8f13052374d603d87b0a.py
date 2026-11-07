"""Capacitated facility location solver using Lagrangian relaxation + subgradient + local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _lagrangian_subproblem(lam, demands, capacities, open_cost, assign_cost):
    """Solve the Lagrangian relaxation subproblem.
    For each customer i, assign to facility f minimizing:
        assign_cost[i,f] + lam[f] * demands[i]
    Also determine which facilities are "open" (assigned at least one customer).
    Returns: (assignment, dual_objective, open_facilities_mask)
    """
    n = len(demands)
    m = len(capacities)
    
    # Modified cost matrix: assign_cost[i,f] + lam[f] * demands[i]
    mod_cost = assign_cost + np.outer(demands, lam)
    
    # For each customer, pick the facility with minimum modified cost
    assignment = np.argmin(mod_cost, axis=1)
    
    # Determine which facilities are open
    open_mask = np.zeros(m, dtype=bool)
    for f in range(m):
        if np.any(assignment == f):
            open_mask[f] = True
    
    # Dual objective: sum of min costs + sum of open_cost for open facilities
    min_costs = mod_cost[np.arange(n), assignment]
    dual_obj = np.sum(min_costs) + np.sum(open_cost[open_mask])
    
    return assignment, dual_obj, open_mask


def _subgradient_lagrangian(demands, capacities, open_cost, assign_cost, max_iter=200, deadline=None):
    """Use subgradient method to find good Lagrange multipliers.
    Returns the best feasible assignment found during the process.
    """
    n = len(demands)
    m = len(capacities)
    
    # Initialize multipliers
    lam = np.zeros(m)
    
    best_dual_obj = np.inf
    best_assignment = None
    best_feasible = None
    best_feasible_cost = np.inf
    
    # Step size parameters
    alpha = 1.0
    best_dual_so_far = np.inf
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Solve subproblem
        assignment, dual_obj, open_mask = _lagrangian_subproblem(lam, demands, capacities, open_cost, assign_cost)
        
        # Track best dual objective
        if dual_obj < best_dual_obj:
            best_dual_obj = dual_obj
            best_assignment = assignment.copy()
        
        # Check feasibility of this assignment
        load = np.zeros(m)
        for i in range(n):
            load[assignment[i]] += demands[i]
        
        if np.all(load <= capacities + 1e-9):
            # Compute actual cost
            actual_cost = np.sum(assign_cost[np.arange(n), assignment]) + np.sum(open_cost[load > 0])
            if actual_cost < best_feasible_cost:
                best_feasible_cost = actual_cost
                best_feasible = assignment.copy()
        
        # Compute subgradient: load[f] - capacities[f] for each f
        subgrad = load - capacities
        
        # Polyak step size if we have a feasible solution
        if best_feasible is not None:
            step = alpha * (best_feasible_cost - dual_obj) / (np.sum(subgrad**2) + 1e-12)
        else:
            # Use diminishing step size
            step = alpha / (it + 1)
        
        # Update multipliers (project onto non-negative orthant)
        lam = np.maximum(0, lam + step * subgrad)
        
        # Decay alpha slowly
        alpha = max(0.1, alpha * 0.999)
    
    # If no feasible solution found, use the best dual assignment and fix it
    if best_feasible is None:
        if best_assignment is not None:
            best_feasible = _fix_capacity_violations(best_assignment, demands, capacities, assign_cost, open_cost)
        else:
            best_feasible = _greedy_assign(demands, capacities, open_cost, assign_cost)
    
    return best_feasible


def _compute_cost(result, demands, capacities, open_cost, assign_cost):
    """Compute total cost of an assignment."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        f = result[i]
        if f < 0 or f >= n_facilities:
            return float('inf')
        load[f] += demands[i]
    
    for f in range(n_facilities):
        if load[f] > capacities[f] + 1e-9:
            return float('inf')
    
    open_cost_total = 0.0
    assign_cost_total = 0.0
    for f in range(n_facilities):
        if load[f] > 0:
            open_cost_total += open_cost[f]
    
    for i in range(n_customers):
        assign_cost_total += assign_cost[i, result[i]]
    
    return open_cost_total + assign_cost_total


def _local_search(result, demands, capacities, open_cost, assign_cost, max_iter=5000):
    """Local search: try moving single customers to other facilities."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    best_result = result.copy()
    best_cost = _compute_cost(best_result, demands, capacities, open_cost, assign_cost)
    
    if not np.isfinite(best_cost):
        return best_result
    
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        load[best_result[i]] += demands[i]
    
    improved = True
    while improved and max_iter > 0:
        improved = False
        max_iter -= 1
        
        for i in range(n_customers):
            current_f = best_result[i]
            current_load_f = load[current_f]
            
            best_alt_f = -1
            best_delta = 0
            
            for f in range(n_facilities):
                if f == current_f:
                    continue
                if load[f] + demands[i] > capacities[f] + 1e-9:
                    continue
                
                delta = -assign_cost[i, current_f] + assign_cost[i, f]
                
                if current_load_f == demands[i]:
                    delta -= open_cost[current_f]
                if load[f] == 0:
                    delta += open_cost[f]
                
                if delta < best_delta - 1e-9:
                    best_delta = delta
                    best_alt_f = f
            
            if best_alt_f >= 0:
                load[current_f] -= demands[i]
                load[best_alt_f] += demands[i]
                best_result[i] = best_alt_f
                best_cost += best_delta
                improved = True
    
    return best_result


def _greedy_assign(demands, capacities, open_cost, assign_cost):
    """Greedy: largest demand first, cheapest feasible facility."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    
    for i in np.argsort(-demands):
        order = np.argsort(assign_cost[i])
        for f in order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
        else:
            remaining = capacities - load
            f = np.argmax(remaining)
            load[f] += demands[i]
            result[i] = f
    return result


def _fix_capacity_violations(result, demands, capacities, assign_cost, open_cost):
    """Fix capacity violations in an assignment by moving customers."""
    n = len(demands)
    m = len(capacities)
    
    result = result.copy()
    load = np.zeros(m)
    for i in range(n):
        load[result[i]] += demands[i]
    
    max_fix_iter = 200
    for _ in range(max_fix_iter):
        violated = [f for f in range(m) if load[f] > capacities[f] + 1e-9]
        if not violated:
            break
        
        worst_f = max(violated, key=lambda f: load[f] - capacities[f])
        customers_at_worst = [i for i in range(n) if result[i] == worst_f]
        
        best_move = None
        best_move_cost = np.inf
        for i in customers_at_worst:
            for f in range(m):
                if f == worst_f:
                    continue
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    move_cost = assign_cost[i, f] - assign_cost[i, worst_f]
                    if load[worst_f] == demands[i]:
                        move_cost -= open_cost[worst_f]
                    if load[f] == 0:
                        move_cost += open_cost[f]
                    if move_cost < best_move_cost:
                        best_move_cost = move_cost
                        best_move = (i, f)
        
        if best_move is None:
            i = customers_at_worst[0]
            remaining = capacities - load
            f = np.argmax(remaining)
            best_move = (i, f)
        
        i, f = best_move
        load[result[i]] -= demands[i]
        load[f] += demands[i]
        result[i] = f
    
    return result


def assign(demands, capacities, open_cost, assign_cost):
    """Assign customers to facilities minimizing total cost."""
    start_time = time.time()
    deadline = start_time + 140
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Phase 1: Lagrangian relaxation with subgradient method
    result = _subgradient_lagrangian(demands, capacities, open_cost, assign_cost, max_iter=300, deadline=deadline - 30)
    
    # Phase 2: Local search to refine
    if time.time() < deadline - 15:
        result = _local_search(result, demands, capacities, open_cost, assign_cost, max_iter=10000)
    
    # Phase 3: Another local search pass
    if time.time() < deadline - 5:
        result = _local_search(result, demands, capacities, open_cost, assign_cost, max_iter=5000)
    
    # Ensure validity
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        f = result[i]
        if f < 0 or f >= n_facilities:
            result = _greedy_assign(demands, capacities, open_cost, assign_cost)
            break
        load[f] += demands[i]
    
    for f in range(n_facilities):
        if load[f] > capacities[f] + 1e-9:
            result = _fix_capacity_violations(result, demands, capacities, assign_cost, open_cost)
            break
    
    return result.astype(int)
# EVOLVE-BLOCK-END
