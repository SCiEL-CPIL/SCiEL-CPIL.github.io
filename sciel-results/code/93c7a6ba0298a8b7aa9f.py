import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Deadline guard: 150s total, leave 10s safety margin
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Helper to compute total cost
    def compute_cost(assignment):
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            f = int(assignment[c])
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

    # 1. Lagrangian Relaxation for a strong initial guess
    def lagrangian_init(demands, capacities, open_cost, assign_cost, deadline):
        n_c = len(demands)
        n_f = len(capacities)
        
        # Initialize dual variables
        mu = np.zeros(n_f)
        
        # Subgradient optimization
        max_iter = 300
        best_assign = None
        
        # Decaying step size
        step_scale = 2.0
        
        for k in range(max_iter):
            if time.time() > deadline - 60:
                break
                
            # 1. Solve Lagrangian subproblem
            # For each customer i, pick f that minimizes c_if + mu_f
            costs = assign_cost + mu[None, :]
            best_f = np.argmin(costs, axis=1)
            
            # 2. Check capacity constraints
            load = np.zeros(n_f)
            for i in range(n_c):
                load[best_f[i]] += demands[i]
            
            # 3. Update mu (Subgradient)
            # Gradient g_f = load_f - capacity_f
            grad = load - capacities
            
            # Norm of gradient
            g_norm_sq = np.sum(grad ** 2)
            if g_norm_sq < 1e-12:
                break
            
            # Update rule: mu_f = max(0, mu_f - step * g_f)
            # Step size: decaying
            step = step_scale * (1.0 + 0.5 * (1.0 - k / max(1, max_iter))) / np.sqrt(g_norm_sq)
            mu = np.maximum(0, mu - step * grad)
        
        # Create feasible assignment from Lagrangian guidance
        # Sort customers by demand (largest first) for better capacity packing
        order = np.argsort(-demands)
        result = np.full(n_c, -1, dtype=int)
        load = np.zeros(n_f)
        
        for i in order:
            costs = assign_cost[i].copy() + mu
            feasible = load + demands[i] <= capacities
            costs[~feasible] = np.inf
            if np.any(np.isfinite(costs)):
                f = int(np.argmin(costs))
            else:
                # Fallback: pick facility with most remaining capacity
                remaining = capacities - load
                f = int(np.argmax(remaining))
            result[i] = f
            load[f] += demands[i]
        
        # Repair any capacity violations
        load = np.zeros(n_f)
        for i in range(n_c):
            load[result[i]] += demands[i]
        
        for _ in range(20):
            if time.time() > deadline - 50:
                break
            violations = [f for f in range(n_f) if load[f] > capacities[f] + 1e-9]
            if not violations:
                break
            
            for f in violations:
                # Find customers assigned to f, move the ones with highest marginal cost to other facilities
                customers_at_f = [i for i in range(n_c) if result[i] == f]
                # Sort by marginal cost of moving (assign_cost to current - assign_cost to best alternative)
                def marginal_cost(i):
                    best_alt = np.inf
                    for f2 in range(n_f):
                        if f2 != f and load[f2] + demands[i] <= capacities[f2]:
                            best_alt = min(best_alt, assign_cost[i, f2])
                    if np.isinf(best_alt):
                        return np.inf
                    return best_alt - assign_cost[i, f]
                
                customers_at_f.sort(key=marginal_cost, reverse=True)
                
                for i in customers_at_f:
                    if load[f] <= capacities[f] + 1e-9:
                        break
                    # Move customer i to best feasible alternative
                    best_f2 = None
                    best_cost = np.inf
                    for f2 in range(n_f):
                        if f2 != f and load[f2] + demands[i] <= capacities[f2]:
                            if assign_cost[i, f2] < best_cost:
                                best_cost = assign_cost[i, f2]
                                best_f2 = f2
                    if best_f2 is not None:
                        load[f] -= demands[i]
                        load[best_f2] += demands[i]
                        result[i] = best_f2
        
        return result

    # 2. Local Search Polish
    def local_search_polish(assignment, demands, capacities, open_cost, assign_cost, deadline):
        n_c = len(demands)
        n_f = len(capacities)
        
        best_assign = assignment.copy()
        best_cost = compute_cost(best_assign)
        
        # Try single reassignments
        for _ in range(100):
            if time.time() > deadline - 15:
                break
            
            improved = False
            for c in range(n_c):
                if time.time() > deadline - 15:
                    break
                current_f = best_assign[c]
                # Try moving to a different facility
                for f2 in range(n_f):
                    if f2 == current_f:
                        continue
                    # Check if move is feasible
                    new_assign = best_assign.copy()
                    new_assign[c] = f2
                    cost = compute_cost(new_assign)
                    if cost < best_cost - 1e-9:
                        best_assign = new_assign
                        best_cost = cost
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                # Try swap of two customers
                swapped = False
                for c1 in range(n_c - 1):
                    if time.time() > deadline - 15:
                        break
                    for c2 in range(c1 + 1, n_c):
                        f1 = best_assign[c1]
                        f2 = best_assign[c2]
                        if f1 == f2:
                            continue
                        # Try swapping
                        new_assign = best_assign.copy()
                        new_assign[c1] = f2
                        new_assign[c2] = f1
                        cost = compute_cost(new_assign)
                        if cost < best_cost - 1e-9:
                            best_assign = new_assign
                            best_cost = cost
                            swapped = True
                            break
                    if swapped:
                        break
                if not swapped:
                    break
        
        return best_assign

    # 3. MIP Formulation
    def solve_mip(demands, capacities, open_cost, assign_cost, time_limit, initial_assign=None):
        n_c = len(demands)
        n_f = len(capacities)
        
        n_vars = n_c * n_f + n_f
        
        # Objective coefficients
        c = np.zeros(n_vars)
        for i in range(n_c):
            for f in range(n_f):
                c[i * n_f + f] = assign_cost[i, f]
        for f in range(n_f):
            c[n_c * n_f + f] = open_cost[f]
        
        # Bounds
        lb = np.zeros(n_vars)
        ub = np.ones(n_vars)
        bounds = Bounds(lb, ub)
        
        # Integrality
        integrality = np.ones(n_vars)
        
        # Constraints Matrix A
        n_rows = n_c + n_f
        A = np.zeros((n_rows, n_vars))
        lb_constr = np.zeros(n_rows)
        ub_constr = np.zeros(n_rows)
        
        # Fill Constraint 1 (Customer assignment)
        for i in range(n_c):
            row_idx = i
            for f in range(n_f):
                col_idx = i * n_f + f
                A[row_idx, col_idx] = 1.0
            lb_constr[row_idx] = 1.0
            ub_constr[row_idx] = 1.0
            
        # Fill Constraint 2 (Capacity)
        for f in range(n_f):
            row_idx = n_c + f
            for i in range(n_c):
                col_idx = i * n_f + f
                A[row_idx, col_idx] = demands[i]
            # Coefficient for y[f]
            col_idx_y = n_c * n_f + f
            A[row_idx, col_idx_y] = -capacities[f]
            lb_constr[row_idx] = -np.inf
            ub_constr[row_idx] = 0.0
            
        constraints = LinearConstraint(A, lb_constr, ub_constr)
        
        try:
            res = milp(c=c,
                       constraints=constraints,
                       integrality=integrality,
                       bounds=bounds,
                       options={'time_limit': time_limit, 'mip_rel_gap': 0.001})
            
            if res.status == 0: # Optimal or feasible
                x_flat = res.x
                # Extract x
                x = x_flat[:n_c * n_f].reshape(n_c, n_f)
                
                # Assign each customer to the facility with x[i,f] closest to 1
                assignment = np.argmax(x, axis=1)
                
                # Verify feasibility
                load = np.zeros(n_f)
                for i in range(n_c):
                    load[assignment[i]] += demands[i]
                
                if np.all(load <= capacities + 1e-6):
                    return assignment
                else:
                    return None
            else:
                return None
                
        except Exception:
            return None

    # Main logic
    # Step 1: Get initial solution from Lagrangian
    init_assign = lagrangian_init(demands, capacities, open_cost, assign_cost, deadline)
    
    # Step 2: Polish with local search
    polished_assign = local_search_polish(init_assign, demands, capacities, open_cost, assign_cost, deadline)
    
    # Step 3: Try MIP with remaining time
    remaining_time = deadline - time.time() - 5.0
    if remaining_time > 10:
        mip_assign = solve_mip(demands, capacities, open_cost, assign_cost, remaining_time)
        if mip_assign is not None:
            # Compare costs
            cost_mip = compute_cost(mip_assign)
            cost_polished = compute_cost(polished_assign)
            if cost_mip < cost_polished:
                final_assign = mip_assign
            else:
                final_assign = polished_assign
        else:
            final_assign = polished_assign
    else:
        final_assign = polished_assign
    
    # Final feasibility check
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        load[final_assign[i]] += demands[i]
    
    if not np.all(load <= capacities + 1e-6):
        # Fallback to greedy
        order = np.argsort(-demands)
        result = np.full(n_customers, -1, dtype=int)
        load = np.zeros(n_facilities)
        
        for i in order:
            fac_order = np.argsort(assign_cost[i])
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break
            else:
                for f in range(n_facilities):
                    if load[f] + demands[i] <= capacities[f]:
                        load[f] += demands[i]
                        result[i] = f
                        break
        final_assign = result
    
    return final_assign
