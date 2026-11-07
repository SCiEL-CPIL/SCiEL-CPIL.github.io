# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csc_matrix


def assign(demands, capacities, open_cost, assign_cost):
    """
    Solve the capacitated facility location problem using MIP with Lagrangian relaxation
    and local search polish.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin out of 150s

    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    n_cust = len(demands)
    n_fac = len(capacities)

    # First try a fast MIP solve
    best_assignment = None
    best_cost = np.inf

    # Try MIP with a time limit
    try:
        result = _solve_mip(demands, capacities, open_cost, assign_cost, start_time, deadline - 5.0)
        if result is not None:
            best_assignment = result
            best_cost = _compute_cost(best_assignment, demands, capacities, open_cost, assign_cost)
    except Exception:
        pass

    # If MIP didn't give a good result, try Lagrangian relaxation + repair
    if best_assignment is None or (time.time() - start_time) < deadline - 10.0:
        lag_assignment = _lagrangian_relaxation(demands, capacities, open_cost, assign_cost, start_time, deadline - 5.0)
        if lag_assignment is not None:
            lag_cost = _compute_cost(lag_assignment, demands, capacities, open_cost, assign_cost)
            if lag_cost < best_cost:
                best_assignment = lag_assignment
                best_cost = lag_cost

    # Local search polish
    if best_assignment is not None and (time.time() - start_time) < deadline - 2.0:
        polished = _local_search(best_assignment, demands, capacities, open_cost, assign_cost, start_time, deadline - 0.5)
        if polished is not None:
            pol_cost = _compute_cost(polished, demands, capacities, open_cost, assign_cost)
            if pol_cost < best_cost:
                best_assignment = polished
                best_cost = pol_cost

    # Final fallback: greedy
    if best_assignment is None:
        best_assignment = _greedy_fallback(demands, capacities, open_cost, assign_cost)

    return best_assignment.astype(int)


def _compute_cost(assignment, demands, capacities, open_cost, assign_cost):
    """Compute the total cost of an assignment."""
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Check validity
    for i in range(n_cust):
        f = assignment[i]
        if f < 0 or f >= n_fac:
            return np.inf
    
    # Check capacity
    load = np.zeros(n_fac, dtype=np.int64)
    for i in range(n_cust):
        load[assignment[i]] += demands[i]
    if np.any(load > capacities):
        return np.inf
    
    # Compute cost
    open_mask = load > 0
    total = np.sum(open_cost[open_mask])
    for i in range(n_cust):
        total += assign_cost[i, assignment[i]]
    return total


def _solve_mip(demands, capacities, open_cost, assign_cost, start_time, deadline):
    """Solve the MIP formulation."""
    n_cust = len(demands)
    n_fac = len(capacities)
    
    n_vars = n_cust * n_fac + n_fac
    
    # Objective
    c = np.zeros(n_vars)
    for i in range(n_cust):
        for f in range(n_fac):
            c[i * n_fac + f] = assign_cost[i, f]
    for f in range(n_fac):
        c[n_cust * n_fac + f] = open_cost[f]
    
    # Constraints
    n_constraints = n_cust + n_fac + n_cust * n_fac
    
    # Build sparse constraint matrix
    rows = []
    cols = []
    vals = []
    lb = np.zeros(n_constraints)
    ub = np.zeros(n_constraints)
    
    row = 0
    # Constraint 1: each customer assigned to exactly one facility
    for i in range(n_cust):
        for f in range(n_fac):
            rows.append(row)
            cols.append(i * n_fac + f)
            vals.append(1.0)
        lb[row] = 1.0
        ub[row] = 1.0
        row += 1
    
    # Constraint 2: capacity constraint
    for f in range(n_fac):
        for i in range(n_cust):
            rows.append(row)
            cols.append(i * n_fac + f)
            vals.append(float(demands[i]))
        rows.append(row)
        cols.append(n_cust * n_fac + f)
        vals.append(-float(capacities[f]))
        lb[row] = -np.inf
        ub[row] = 0.0
        row += 1
    
    # Constraint 3: linking constraint x[i, f] <= y[f]
    for i in range(n_cust):
        for f in range(n_fac):
            rows.append(row)
            cols.append(i * n_fac + f)
            vals.append(1.0)
            rows.append(row)
            cols.append(n_cust * n_fac + f)
            vals.append(-1.0)
            lb[row] = -np.inf
            ub[row] = 0.0
            row += 1
    
    A = csc_matrix((vals, (rows, cols)), shape=(n_constraints, n_vars))
    
    # Variable bounds and integrality
    integrality = np.zeros(n_vars)
    for i in range(n_cust):
        for f in range(n_fac):
            integrality[i * n_fac + f] = 1.0
    for f in range(n_fac):
        integrality[n_cust * n_fac + f] = 1.0
    
    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    constraint = LinearConstraint(A, lb, ub)
    
    # Check time
    if time.time() > deadline:
        return None
    
    # Solve MIP
    result = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
    
    if not result.success:
        return None
    
    # Extract solution
    x = result.x[:n_cust * n_fac].reshape(n_cust, n_fac)
    assignment = np.zeros(n_cust, dtype=int)
    for i in range(n_cust):
        assignment[i] = np.argmax(x[i])
    
    # Verify
    if _compute_cost(assignment, demands, capacities, open_cost, assign_cost) == np.inf:
        return None
    
    return assignment


def _lagrangian_relaxation(demands, capacities, open_cost, assign_cost, start_time, deadline):
    """
    Lagrangian relaxation of capacity constraints.
    Dual variables lambda[f] >= 0 for each facility capacity constraint.
    Relaxed problem decomposes into independent customer assignments.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Initialize dual variables
    lam = np.zeros(n_fac)
    
    best_assignment = None
    best_cost = np.inf
    
    max_iter = 500
    step_size = 1.0
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Solve relaxed problem: for each customer, pick facility minimizing
        # assign_cost[i, f] + lam[f] * demands[i]
        # But we also need to account for open costs. 
        # Actually, the Lagrangian relaxation of capacity constraints gives:
        # min sum_i sum_f (assign_cost[i,f] + lam[f]*demands[i]) x[i,f] + sum_f open_cost[f]*y[f]
        # subject to sum_f x[i,f] = 1, x[i,f] <= y[f]
        # This still has the linking constraint. Let's simplify by ignoring open costs
        # in the relaxation and handling them in repair.
        
        # For each customer, pick the facility that minimizes assign_cost[i,f] + lam[f]*demands[i]
        relaxed_cost = assign_cost + np.outer(demands, lam)
        relaxed_assignment = np.argmin(relaxed_cost, axis=1)
        
        # Compute load
        load = np.zeros(n_fac, dtype=np.int64)
        for i in range(n_cust):
            load[relaxed_assignment[i]] += demands[i]
        
        # Check if feasible
        if np.all(load <= capacities):
            # Feasible! Compute actual cost
            cost = _compute_cost(relaxed_assignment, demands, capacities, open_cost, assign_cost)
            if cost < best_cost:
                best_cost = cost
                best_assignment = relaxed_assignment.copy()
        
        # Update dual variables via subgradient method
        # Subgradient: g[f] = load[f] - capacities[f] (positive if over capacity)
        grad = load - capacities.astype(np.float64)
        
        # Project: lam[f] = max(0, lam[f] - step_size * grad[f])
        # If over capacity, decrease lam; if under, increase lam (but keep non-negative)
        lam = np.maximum(0, lam + step_size * grad)
        
        # Decrease step size
        step_size = max(0.1, step_size * 0.98)
        
        # Also try a greedy repair of the relaxed assignment
        if time.time() < deadline - 1.0:
            repaired = _repair_capacity_violations(relaxed_assignment, demands, capacities, open_cost, assign_cost)
            if repaired is not None:
                rep_cost = _compute_cost(repaired, demands, capacities, open_cost, assign_cost)
                if rep_cost < best_cost:
                    best_cost = rep_cost
                    best_assignment = repaired
    
    return best_assignment


def _repair_capacity_violations(assignment, demands, capacities, open_cost, assign_cost):
    """Repair capacity violations by reassigning customers."""
    n_cust = len(demands)
    n_fac = len(capacities)
    
    load = np.zeros(n_fac, dtype=np.int64)
    for i in range(n_cust):
        load[assignment[i]] += demands[i]
    
    # Find over-capacity facilities
    for f in range(n_fac):
        if load[f] > capacities[f]:
            # Need to move some customers out of facility f
            excess = load[f] - capacities[f]
            
            # Find customers assigned to f, sorted by how much worse it would be to move them
            customers_at_f = [i for i in range(n_cust) if assignment[i] == f]
            
            # For each customer at f, compute the best alternative
            moves = []
            for i in customers_at_f:
                best_alt_cost = np.inf
                best_alt_fac = -1
                for f2 in range(n_fac):
                    if f2 == f:
                        continue
                    if load[f2] + demands[i] <= capacities[f2]:
                        alt_cost = assign_cost[i, f2]
                        if alt_cost < best_alt_cost:
                            best_alt_cost = alt_cost
                            best_alt_fac = f2
                if best_alt_fac >= 0:
                    moves.append((best_alt_cost, i, best_alt_fac))
            
            # Sort by cost of moving (cheapest first)
            moves.sort()
            
            for alt_cost, i, alt_fac in moves:
                if excess <= 0:
                    break
                # Move customer i to alt_fac
                load[assignment[i]] -= demands[i]
                load[alt_fac] += demands[i]
                assignment[i] = alt_fac
                excess -= demands[i]
    
    # Verify
    for i in range(n_cust):
        if assignment[i] < 0 or assignment[i] >= n_fac:
            return None
    for f in range(n_fac):
        if load[f] > capacities[f]:
            return None
    
    return assignment


def _local_search(assignment, demands, capacities, open_cost, assign_cost, start_time, deadline):
    """Local search: try moving or swapping customers to improve cost."""
    n_cust = len(demands)
    n_fac = len(capacities)
    
    current_cost = _compute_cost(assignment, demands, capacities, open_cost, assign_cost)
    if current_cost == np.inf:
        return None
    
    load = np.zeros(n_fac, dtype=np.int64)
    for i in range(n_cust):
        load[assignment[i]] += demands[i]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try single moves
        for i in range(n_cust):
            if time.time() > deadline:
                break
            current_f = assignment[i]
            for f2 in range(n_fac):
                if f2 == current_f:
                    continue
                if load[f2] + demands[i] <= capacities[f2]:
                    # Compute cost change
                    delta = (assign_cost[i, f2] - assign_cost[i, current_f])
                    # Check if opening f2 adds cost
                    if load[f2] == 0:
                        delta += open_cost[f2]
                    # Check if closing current_f saves cost
                    if load[current_f] == demands[i]:
                        delta -= open_cost[current_f]
                    
                    if delta < -1e-9:
                        # Make the move
                        load[current_f] -= demands[i]
                        load[f2] += demands[i]
                        assignment[i] = f2
                        current_cost += delta
                        improved = True
                        break
        
        # Try swaps
        if not improved:
            for i in range(n_cust):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n_cust):
                    if time.time() > deadline:
                        break
                    fi = assignment[i]
                    fj = assignment[j]
                    if fi == fj:
                        continue
                    
                    # Check if swapping is feasible
                    if (load[fi] - demands[i] + demands[j] <= capacities[fi] and
                        load[fj] - demands[j] + demands[i] <= capacities[fj]):
                        # Compute cost change
                        delta = (assign_cost[i, fj] + assign_cost[j, fi] -
                                 assign_cost[i, fi] - assign_cost[j, fj])
                        
                        # Open/close cost changes
                        if load[fi] == demands[i]:
                            # fi would close if i moves
                            delta -= open_cost[fi]
                            if load[fj] - demands[j] + demands[i] > 0:
                                # fj stays open
                                pass
                        if load[fj] == demands[j]:
                            # fj would close if j moves
                            delta -= open_cost[fj]
                        
                        if load[fi] - demands[i] + demands[j] == 0 and load[fi] != demands[i]:
                            # fi closes
                            delta -= open_cost[fi]
                        if load[fj] - demands[j] + demands[i] == 0 and load[fj] != demands[j]:
                            # fj closes
                            delta -= open_cost[fj]
                        
                        if delta < -1e-9:
                            # Make the swap
                            load[fi] -= demands[i]
                            load[fj] -= demands[j]
                            load[fi] += demands[j]
                            load[fj] += demands[i]
                            assignment[i] = fj
                            assignment[j] = fi
                            current_cost += delta
                            improved = True
                            break
                if improved:
                    break
    
    return assignment


def _greedy_fallback(demands, capacities, open_cost, assign_cost):
    """Greedy fallback: largest demand first, cheapest feasible facility."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    for i in np.argsort(-demands):
        best_f = -1
        best_cost = np.inf
        for f in np.argsort(assign_cost[i]):
            if load[f] + demands[i] <= capacities[f]:
                # Account for opening cost if this is the first customer at f
                if load[f] == 0:
                    total = assign_cost[i, f] + open_cost[f]
                else:
                    total = assign_cost[i, f]
                if total < best_cost:
                    best_cost = total
                    best_f = f
        if best_f >= 0:
            load[best_f] += demands[i]
            result[i] = best_f
        else:
            # Should not happen if problem is feasible
            result[i] = 0
    return result
# EVOLVE-BLOCK-END
