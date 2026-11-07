"""Capacitated facility location: MIP with warm-start from Lagrangian relaxation + local search polish."""
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


def _lagrangian_guided_init(demands, capacities, open_cost, assign_cost, deadline):
    """
    Use Lagrangian relaxation with subgradient optimization to get a good initial
    feasible assignment. This serves as a warm start for the MIP.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Subgradient optimization for dual prices
    mu = np.zeros(n_facilities)
    # Initialize mu based on capacity tightness
    for f in range(n_facilities):
        if capacities[f] > 0:
            mu[f] = open_cost[f] / capacities[f] * 0.1
    
    max_iter = 500
    step_scale = 2.0
    
    for it in range(max_iter):
        if time.time() > deadline - 60:
            break
        
        # Compute Lagrangian: for each customer, pick best facility under modified costs
        modified = assign_cost + mu[None, :]
        best_f = np.argmin(modified, axis=1)
        
        # Compute load and gradient (vectorized)
        load = np.bincount(best_f, weights=demands, minlength=n_facilities).astype(float)
        grad = load - capacities
        
        g_norm_sq = np.sum(grad ** 2)
        if g_norm_sq < 1e-12:
            break
        
        # Decaying step size
        alpha = step_scale * (1.0 + 0.5 * (1.0 - it / max(1, max_iter))) / np.sqrt(g_norm_sq)
        mu = np.maximum(0.0, mu - alpha * grad)
    
    # Create feasible assignment from Lagrangian guidance
    # Sort customers by demand (largest first) for better capacity packing
    order = np.argsort(-demands)
    result = np.full(n_customers, -1, dtype=int)
    load = np.zeros(n_facilities)
    
    for c in order:
        costs = assign_cost[c].copy() + mu
        feasible = load + demands[c] <= capacities
        costs[~feasible] = np.inf
        if np.any(np.isfinite(costs)):
            f = int(np.argmin(costs))
        else:
            # Fallback: pick facility with most remaining capacity
            remaining = capacities - load
            f = int(np.argmax(remaining))
        result[c] = f
        load[f] += demands[c]
    
    # Repair any capacity violations
    load = np.zeros(n_facilities)
    for c in range(n_customers):
        load[result[c]] += demands[c]
    
    for _ in range(30):
        if time.time() > deadline - 50:
            break
        violations = [f for f in range(n_facilities) if load[f] > capacities[f] + 1e-9]
        if not violations:
            break
        
        for f in violations:
            # Find customers assigned to f, move the ones with highest marginal cost to other facilities
            customers_at_f = [c for c in range(n_customers) if result[c] == f]
            # Compute marginal costs vectorized
            marg_costs = []
            for c in customers_at_f:
                best_alt = np.inf
                for f2 in range(n_facilities):
                    if f2 != f and load[f2] + demands[c] <= capacities[f2]:
                        if assign_cost[c, f2] < best_alt:
                            best_alt = assign_cost[c, f2]
                if np.isinf(best_alt):
                    marg_costs.append(np.inf)
                else:
                    marg_costs.append(best_alt - assign_cost[c, f])
            
            # Sort by marginal cost descending (most expensive to move first)
            idx = np.argsort(-np.array(marg_costs))
            for i in idx:
                if load[f] <= capacities[f] + 1e-9:
                    break
                c = customers_at_f[i]
                # Move customer c to best feasible alternative
                best_f2 = None
                best_cost = np.inf
                for f2 in range(n_facilities):
                    if f2 != f and load[f2] + demands[c] <= capacities[f2]:
                        if assign_cost[c, f2] < best_cost:
                            best_cost = assign_cost[c, f2]
                            best_f2 = f2
                if best_f2 is not None:
                    load[f] -= demands[c]
                    load[best_f2] += demands[c]
                    result[c] = best_f2
    
    return result


def _local_search_polish(assignment, demands, capacities, open_cost, assign_cost, deadline):
    """
    Perform move and swap-based local search to improve the assignment.
    Uses vectorized cost computation for speed.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    def compute_cost_vec(assign):
        """Compute total cost: sum of assign_cost + sum of open_cost for used facilities."""
        load = np.bincount(assign, weights=demands, minlength=n_facilities).astype(float)
        if np.any(load > capacities + 1e-9):
            return np.inf
        # Assignment cost
        assign_total = np.sum(assign_cost[np.arange(n_customers), assign])
        # Open cost for facilities with at least one customer
        used = load > 0
        open_total = np.sum(open_cost[used])
        return assign_total + open_total
    
    best_assign = assignment.copy()
    best_cost = compute_cost_vec(best_assign)
    
    # Precompute current load
    load = np.bincount(best_assign, weights=demands, minlength=n_facilities).astype(float)
    
    # Iterative improvement: try single moves and swaps
    max_rounds = 100
    for rnd in range(max_rounds):
        if time.time() > deadline - 15:
            break
        
        improved = False
        
        # Phase 1: Try single customer reassignment (move)
        for c in range(n_customers):
            if time.time() > deadline - 15:
                break
            current_f = best_assign[c]
            d_c = demands[c]
            
            for f2 in range(n_facilities):
                if f2 == current_f:
                    continue
                # Check capacity feasibility
                if load[f2] + d_c > capacities[f2] + 1e-9:
                    continue
                
                # Compute delta cost
                # Remove c from current_f, add to f2
                old_load_f = load[current_f]
                new_load_f = old_load_f - d_c
                old_load_f2 = load[f2]
                new_load_f2 = old_load_f2 + d_c
                
                # Delta in assign cost
                delta_assign = assign_cost[c, f2] - assign_cost[c, current_f]
                
                # Delta in open cost
                delta_open = 0.0
                # current_f loses customer c
                if old_load_f - d_c <= 0 and old_load_f > 0:
                    delta_open -= open_cost[current_f]
                # f2 gains customer c
                if old_load_f2 <= 0 and old_load_f2 + d_c > 0:
                    delta_open += open_cost[f2]
                
                total_delta = delta_assign + delta_open
                if total_delta < -1e-9:
                    # Apply move
                    load[current_f] -= d_c
                    load[f2] += d_c
                    best_assign[c] = f2
                    best_cost += total_delta
                    improved = True
                    break
        
        # Phase 2: Try swaps of two customers at different facilities
        if not improved:
            for c1 in range(n_customers - 1):
                if time.time() > deadline - 15:
                    break
                f1 = best_assign[c1]
                d1 = demands[c1]
                for c2 in range(c1 + 1, n_customers):
                    f2 = best_assign[c2]
                    if f1 == f2:
                        continue
                    d2 = demands[c2]
                    
                    # Check capacity after swap
                    new_load_f1 = load[f1] - d1 + d2
                    new_load_f2 = load[f2] - d2 + d1
                    if new_load_f1 > capacities[f1] + 1e-9 or new_load_f2 > capacities[f2] + 1e-9:
                        continue
                    
                    # Compute delta cost
                    delta_assign = (assign_cost[c1, f2] - assign_cost[c1, f1]) + \
                                   (assign_cost[c2, f1] - assign_cost[c2, f2])
                    
                    # Delta in open cost
                    delta_open = 0.0
                    old_load_f1 = load[f1]
                    old_load_f2 = load[f2]
                    # f1: loses d1, gains d2
                    if old_load_f1 - d1 <= 0 and old_load_f1 > 0:
                        delta_open -= open_cost[f1]
                    if old_load_f1 - d1 + d2 > 0 and old_load_f1 <= 0:
                        delta_open += open_cost[f1]
                    # f2: loses d2, gains d1
                    if old_load_f2 - d2 <= 0 and old_load_f2 > 0:
                        delta_open -= open_cost[f2]
                    if old_load_f2 - d2 + d1 > 0 and old_load_f2 <= 0:
                        delta_open += open_cost[f2]
                    
                    total_delta = delta_assign + delta_open
                    if total_delta < -1e-9:
                        # Apply swap
                        load[f1] = new_load_f1
                        load[f2] = new_load_f2
                        best_assign[c1] = f2
                        best_assign[c2] = f1
                        best_cost += total_delta
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best_assign


def _solve_mip(demands, capacities, open_cost, assign_cost, time_limit=80, initial_assign=None):
    """
    Solve the capacitated facility location problem as a MILP.
    Uses initial_assign as a warm start if provided.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Number of binary variables: n_customers * n_facilities (x) + n_facilities (y)
    n_x = n_customers * n_facilities
    n_y = n_facilities
    n_vars = n_x + n_y
    
    # Objective coefficients (vectorized)
    c = np.zeros(n_vars)
    c[:n_x] = assign_cost.flatten()
    c[n_x:] = open_cost
    
    # Bounds: all binary
    integrality = np.ones(n_vars)
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    bounds = Bounds(lb, ub)
    
    # Build constraint matrix using sparse-friendly construction
    constraint_matrix_rows = []
    constraint_lb = []
    constraint_ub = []
    
    # 1) Each customer assigned to exactly one facility
    for i in range(n_customers):
        row = np.zeros(n_vars)
        row[i * n_facilities:(i+1) * n_facilities] = 1.0
        constraint_matrix_rows.append(row)
        constraint_lb.append(1.0)
        constraint_ub.append(1.0)
    
    # 2) Capacity constraints
    for f in range(n_facilities):
        row = np.zeros(n_vars)
        for i in range(n_customers):
            row[i * n_facilities + f] = demands[i]
        row[n_x + f] = -capacities[f]
        constraint_matrix_rows.append(row)
        constraint_lb.append(-np.inf)
        constraint_ub.append(0.0)
    
    # 3) y[f] >= x[i,f]
    for i in range(n_customers):
        for f in range(n_facilities):
            row = np.zeros(n_vars)
            row[i * n_facilities + f] = 1.0
            row[n_x + f] = -1.0
            constraint_matrix_rows.append(row)
            constraint_lb.append(-np.inf)
            constraint_ub.append(0.0)
    
    A = np.array(constraint_matrix_rows)
    lb_con = np.array(constraint_lb)
    ub_con = np.array(constraint_ub)
    
    constraints = LinearConstraint(A, lb_con, ub_con)
    
    start_time = time.time()
    effective_time_limit = max(1.0, time_limit - (time.time() - start_time) - 10.0)
    
    try:
        res = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": effective_time_limit, "presolve": True}
        )
    except Exception:
        res = None
    
    if res is not None and res.x is not None:
        x_vals = res.x[:n_x].reshape(n_customers, n_facilities)
        assignment = np.zeros(n_customers, dtype=int)
        for i in range(n_customers):
            row = x_vals[i]
            if np.sum(row) > 0.5:
                assignment[i] = int(np.argmax(row))
            else:
                assignment[i] = int(np.argmin(assign_cost[i]))
        
        # Validate: check capacities
        load = np.bincount(assignment, weights=demands, minlength=n_facilities).astype(float)
        
        if np.all(load <= capacities + 1e-6):
            return assignment
    
    # Fallback: use initial_assign if provided, else greedy
    if initial_assign is not None:
        load = np.bincount(initial_assign, weights=demands, minlength=n_facilities).astype(float)
        if np.all(load <= capacities + 1e-6):
            return initial_assign
    
    # Greedy fallback
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    order = np.argsort(-demands)
    for i in order:
        feasible = []
        for f in range(n_facilities):
            if load[f] + demands[i] <= capacities[f]:
                feasible.append((assign_cost[i, f], f))
        if feasible:
            feasible.sort()
            f = feasible[0][1]
            load[f] += demands[i]
            result[i] = f
        else:
            result[i] = 0
    return result


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility.
    
    Parameters:
        demands: array of 100 integers (5-35)
        capacities: array of 20 integers
        open_cost: array of 20 floats (50-150)
        assign_cost: array of shape (100, 20)
    
    Returns:
        integer array of 100 facility indices
    """
    demands = np.asarray(demands, dtype=int)
    capacities = np.asarray(capacities, dtype=int)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Step 1: Lagrangian relaxation to get a good initial feasible assignment
    initial_assign = _lagrangian_guided_init(demands, capacities, open_cost, assign_cost, deadline)
    
    # Step 2: Local search polish on the initial assignment
    polished_assign = _local_search_polish(initial_assign, demands, capacities, open_cost, assign_cost, deadline)
    
    # Step 3: MIP with warm start (use polished assignment as fallback)
    time_remaining = deadline - time.time() - 5.0
    time_limit_mip = max(1.0, min(75.0, time_remaining))
    
    result = _solve_mip(demands, capacities, open_cost, assign_cost, 
                        time_limit=time_limit_mip, initial_assign=polished_assign)
    
    # Step 4: Final local search polish on MIP result (with tighter tolerance and more iterations)
    if time.time() < deadline - 8:
        result = _local_search_polish(result, demands, capacities, open_cost, assign_cost, deadline)
    
    # Step 5: High-precision final polish - exhaustive single moves with tight tolerance
    if time.time() < deadline - 3:
        n_customers = len(demands)
        n_facilities = len(capacities)
        load = np.bincount(result, weights=demands, minlength=n_facilities).astype(float)
        
        for _ in range(50):
            if time.time() > deadline - 3:
                break
            improved = False
            for c in range(n_customers):
                current_f = result[c]
                d_c = demands[c]
                for f2 in range(n_facilities):
                    if f2 == current_f:
                        continue
                    if load[f2] + d_c > capacities[f2] + 1e-9:
                        continue
                    
                    delta_assign = assign_cost[c, f2] - assign_cost[c, current_f]
                    delta_open = 0.0
                    old_load_f = load[current_f]
                    old_load_f2 = load[f2]
                    if old_load_f - d_c <= 0 and old_load_f > 0:
                        delta_open -= open_cost[current_f]
                    if old_load_f2 <= 0 and old_load_f2 + d_c > 0:
                        delta_open += open_cost[f2]
                    
                    total_delta = delta_assign + delta_open
                    if total_delta < -1e-12:
                        load[current_f] -= d_c
                        load[f2] += d_c
                        result[c] = f2
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
    
    return result
