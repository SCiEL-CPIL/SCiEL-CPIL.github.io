"""Rewritten program using Lagrangian Relaxation for Capacitated Facility Location."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def _solve_ufl_lp(assign_cost_mod, open_cost):
    """
    Solves the LP relaxation of the Uncapacitated Facility Location Problem.
    Min sum_{i,j} c_ij x_ij + sum_j F_j y_j
    s.t. sum_j x_ij = 1 for all i
         x_ij <= y_j for all i, j
         x_ij >= 0, y_j >= 0
    """
    n_cust = assign_cost_mod.shape[0]
    n_fac = assign_cost_mod.shape[1]
    
    # Variables: x_0...x_{n_cust*n_fac-1}, y_0...y_{n_fac-1}
    # Total variables: n_cust * n_fac + n_fac
    n_vars = n_cust * n_fac + n_fac
    
    # Objective coefficients
    c_obj = np.zeros(n_vars)
    for i in range(n_cust):
        for j in range(n_fac):
            c_obj[i * n_fac + j] = assign_cost_mod[i, j]
    for j in range(n_fac):
        c_obj[n_cust * n_fac + j] = open_cost[j]
        
    # Constraints
    # 1. sum_j x_ij = 1 for each i
    #    Matrix A_eq: n_cust rows, n_vars cols
    A_eq = np.zeros((n_cust, n_vars))
    b_eq = np.ones(n_cust)
    for i in range(n_cust):
        for j in range(n_fac):
            A_eq[i, i * n_fac + j] = 1.0
            
    # 2. x_ij - y_j <= 0 for each i, j
    #    Matrix A_ub: n_cust * n_fac rows, n_vars cols
    A_ub = np.zeros((n_cust * n_fac, n_vars))
    b_ub = np.zeros(n_cust * n_fac)
    for i in range(n_cust):
        for j in range(n_fac):
            idx = i * n_fac + j
            A_ub[idx, i * n_fac + j] = 1.0
            A_ub[idx, n_cust * n_fac + j] = -1.0
            
    # Bounds: 0 <= x, y <= 1
    bounds = [(0, 1)] * n_vars
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.status == 0:
            x_flat = res.x[:n_cust * n_fac]
            y_vec = res.x[n_cust * n_fac:]
            x_mat = x_flat.reshape(n_cust, n_fac)
            return x_mat, y_vec
        else:
            return None, None
    except Exception:
        return None, None

def _round_and_repair(x_frac, y_frac, demands, capacities, open_cost, assign_cost):
    """
    Rounds the fractional UFL solution to a feasible CFLP solution.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Determine which facilities to open
    # Threshold 0.5 is standard, but we can tune this
    open_facilities = np.where(y_frac > 0.5)[0]
    
    # If no facilities are open, open the one with lowest open_cost
    if len(open_facilities) == 0:
        open_facilities = np.array([np.argmin(open_cost)])
        
    # Initial assignment: assign each customer to the open facility with lowest cost
    # We restrict assignment_cost to only open facilities
    # Create a mask for open facilities
    mask = np.zeros(n_fac, dtype=bool)
    mask[open_facilities] = True
    
    # Set costs for closed facilities to infinity
    cost_masked = assign_cost.copy()
    cost_masked[:, ~mask] = np.inf
    
    # Assign each customer to the facility with minimum cost among open ones
    result = np.argmin(cost_masked, axis=1)
    
    # Check capacity and repair
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
        
    # Repair loop: move customers from overloaded facilities to others
    # We prioritize moving customers with highest demand or lowest "penalty" for moving
    # Simple greedy repair:
    # Find overloaded facilities
    overloaded = np.where(load > capacities)[0]
    
    max_iter = 1000
    for _ in range(max_iter):
        if len(overloaded) == 0:
            break
            
        # Pick the most overloaded facility
        f_over = overloaded[np.argmax(load[overloaded] - capacities[overloaded])]
        
        # Find customers assigned to f_over
        custs_in_over = np.where(result == f_over)[0]
        
        # Sort these customers by demand (descending) to move large ones first? 
        # Or by "regret" (difference between next best and current)?
        # Let's sort by demand descending to free up space quickly
        custs_in_over = custs_in_over[np.argsort(-demands[custs_in_over])]
        
        moved = False
        for i in custs_in_over:
            # Find the best alternative facility
            # Alternative must have space: load[f] + demands[i] <= capacities[f]
            # And f != f_over
            best_f = -1
            best_cost = np.inf
            
            for f in range(n_fac):
                if f == f_over:
                    continue
                if load[f] + demands[i] <= capacities[f]:
                    # Cost to move i to f
                    # Note: We don't change open_cost here as f is already open or we might need to open it?
                    # In this repair, we assume we can only move to already open facilities 
                    # or we allow opening new ones if beneficial? 
                    # For simplicity and validity, let's only move to already open facilities.
                    # If no space in open facilities, we must open a new one.
                    # But opening a new one changes the structure. 
                    # Let's stick to moving to open facilities first.
                    c_move = assign_cost[i, f]
                    if c_move < best_cost:
                        best_cost = c_move
                        best_f = f
                        
            if best_f == -1:
                # No space in open facilities. Must open a new one.
                # Find the facility with most remaining capacity (or lowest cost to open + assign)
                # We will open the facility that minimizes open_cost[f] + assign_cost[i, f] 
                # subject to capacities[f] >= demands[i]
                candidates = np.where(capacities >= demands[i])[0]
                if len(candidates) == 0:
                    # Infeasible instance? Should not happen with valid inputs.
                    # Fallback: assign to any facility with space, even if we ignore capacity (invalid)
                    # But we must return valid. 
                    # Let's just pick the one with most space.
                    f_new = np.argmax(capacities - load)
                    if load[f_new] + demands[i] > capacities[f_new]:
                         # Truly infeasible or stuck. 
                         # Break and hope previous state was valid? 
                         # Actually, if we can't place a customer, the problem is infeasible.
                         # But the task guarantees solvable instances.
                         pass
                    else:
                        # Open f_new
                        mask[f_new] = True
                        load[f_new] += demands[i]
                        load[f_over] -= demands[i]
                        result[i] = f_new
                        moved = True
                        # Update overloaded list
                        overloaded = np.where(load > capacities)[0]
                        break
                else:
                    # Pick best candidate
                    costs = open_cost[candidates] + assign_cost[i, candidates]
                    # Also consider that we are opening a new facility
                    # The cost impact is open_cost[f] + assign_cost[i, f] - assign_cost[i, f_over]
                    # But we are just repairing feasibility, not optimizing cost strictly here.
                    # Let's pick the one with lowest assign_cost[i, f] among candidates
                    # to minimize assignment cost increase.
                    idx_min = np.argmin(assign_cost[i, candidates])
                    f_new = candidates[idx_min]
                    
                    # Open f_new
                    mask[f_new] = True
                    load[f_new] += demands[i]
                    load[f_over] -= demands[i]
                    result[i] = f_new
                    moved = True
                    # Update overloaded list
                    overloaded = np.where(load > capacities)[0]
                    break
                    
            if moved:
                continue
            
            # If no customer could be moved (should not happen if space exists elsewhere)
            # Force move the first one to the facility with most space
            f_target = np.argmax(capacities - load)
            if f_target != f_over and load[f_target] + demands[i] <= capacities[f_target]:
                load[f_target] += demands[i]
                load[f_over] -= demands[i]
                result[i] = f_target
                moved = True
                overloaded = np.where(load > capacities)[0]
                break
            elif f_target == f_over:
                # All facilities are f_over? Impossible.
                break
            else:
                # Try next customer
                continue
                
        if not moved:
            break
            
    return result

def _compute_cost(result, demands, capacities, open_cost, assign_cost):
    """Compute total cost of an assignment."""
    n_cust = len(demands)
    n_fac = len(capacities)
    if len(result) != n_cust:
        return np.inf
    for i in range(n_cust):
        if result[i] < 0 or result[i] >= n_fac:
            return np.inf
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
    for f in range(n_fac):
        if load[f] > capacities[f]:
            return np.inf
    total = 0.0
    for f in range(n_fac):
        if load[f] > 0:
            total += open_cost[f]
    for i in range(n_cust):
        total += assign_cost[i, result[i]]
    return total

def _local_search(demands, capacities, open_cost, assign_cost, result, time_limit=10.0):
    """Simple local search to polish the solution."""
    start_time = time.time()
    n_cust = len(demands)
    n_fac = len(capacities)
    result = result.copy()
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
    
    best_cost = _compute_cost(result, demands, capacities, open_cost, assign_cost)
    if np.isinf(best_cost):
        return result
    
    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        for i in range(n_cust):
            if time.time() - start_time > time_limit:
                break
            current_f = result[i]
            best_delta = 0
            best_f = current_f
            for f in range(n_fac):
                if f == current_f:
                    continue
                if load[f] + demands[i] > capacities[f]:
                    continue
                delta = assign_cost[i, f] - assign_cost[i, current_f]
                if load[current_f] == demands[i]:
                    delta -= open_cost[current_f]
                if load[f] == 0:
                    delta += open_cost[f]
                if delta < best_delta:
                    best_delta = delta
                    best_f = f
            if best_delta < -1e-9:
                load[current_f] -= demands[i]
                load[best_f] += demands[i]
                result[i] = best_f
                best_cost += best_delta
                improved = True
    return result

def assign(demands, capacities, open_cost, assign_cost):
    """Assign customers to facilities using Lagrangian Relaxation."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Initial solution via Greedy
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    order = np.argsort(-demands)
    for i in order:
        assigned = False
        for f in np.argsort(assign_cost[i]):
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                assigned = True
                break
        if not assigned:
            rem = capacities - load
            f = np.argmax(rem)
            load[f] += demands[i]
            result[i] = f
            
    best_result = result.copy()
    best_cost = _compute_cost(best_result, demands, capacities, open_cost, assign_cost)
    
    # Lagrangian Relaxation Loop
    lambda_vec = np.zeros(n_fac)
    gap = 1.0
    prev_bound = np.inf
    stagnation_count = 0
    
    while time.time() < deadline - 20.0:
        # Compute modified costs
        # c'_ij = c_ij + lambda_j * d_i
        # We can do this efficiently
        c_prime = assign_cost + np.outer(demands, lambda_vec)
        
        # Solve UFL LP
        x_frac, y_frac = _solve_ufl_lp(c_prime, open_cost)
        
        if x_frac is None:
            break
            
        # Compute Lagrangian Bound
        # L(lambda) = sum c'_ij x_ij + sum F_j y_j - sum lambda_j C_j
        lp_obj = np.sum(c_prime * x_frac) + np.dot(open_cost, y_frac)
        bound = lp_obj - np.dot(lambda_vec, capacities)
        
        # Update Gap
        if bound > best_cost:
            gap = bound - best_cost
        else:
            gap = 0.0
            
        if abs(prev_bound - bound) < 1e-5 and stagnation_count > 10:
             # Stagnation, maybe break or perturb
             pass
        else:
             stagnation_count = 0
        prev_bound = bound
        
        # Round and Repair
        candidate = _round_and_repair(x_frac, y_frac, demands, capacities, open_cost, assign_cost)
        
        # Polish with local search (short time)
        elapsed = time.time() - start_time
        time_left = deadline - 20.0 - elapsed
        if time_left > 1.0:
            candidate = _local_search(demands, capacities, open_cost, assign_cost, candidate, time_limit=min(5.0, time_left))
            
        cand_cost = _compute_cost(candidate, demands, capacities, open_cost, assign_cost)
        
        if cand_cost < best_cost:
            best_cost = cand_cost
            best_result = candidate.copy()
            
        # Update Lagrange Multipliers
        # Subgradient: g_j = L_j - C_j
        load_cand = np.zeros(n_fac)
        for i in range(n_cust):
            load_cand[candidate[i]] += demands[i]
            
        subgrad = load_cand - capacities
        
        # Step size
        if np.sum(subgrad**2) > 1e-10:
            alpha = 0.5 * gap / np.sum(subgrad**2)
        else:
            alpha = 0.0
            
        lambda_vec = np.maximum(0, lambda_vec + alpha * subgrad)
        
        # Optional: Decay alpha if stuck
        # For simplicity, we keep it as is.
        
    # Final Polish on Best Result
    elapsed = time.time() - start_time
    time_left = deadline - 5.0 - elapsed
    if time_left > 1.0:
        best_result = _local_search(demands, capacities, open_cost, assign_cost, best_result, time_limit=time_left)
        
    return best_result
# EVOLVE-BLOCK-END
