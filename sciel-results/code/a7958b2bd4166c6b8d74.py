import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds
import warnings

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 140.0
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    def calc_cost(a):
        """Calculate total cost of assignment."""
        cost = 0.0
        for f in range(n_facilities):
            if np.any(a == f):
                cost += open_cost[f]
        cost += np.sum(assign_cost[np.arange(n_customers), a])
        return cost
    
    def local_search(assignment, time_budget=10.0):
        """Local search: try moving each customer to a better facility."""
        assignment = assignment.copy()
        load = np.zeros(n_facilities)
        for i in range(n_customers):
            load[assignment[i]] += demands[i]
        
        improved = True
        while improved and time.time() < start_time + time_budget:
            improved = False
            for i in range(n_customers):
                curr_f = assignment[i]
                curr_load = load[curr_f]
                
                best_delta = 0.0
                best_f = curr_f
                
                for new_f in range(n_facilities):
                    if new_f == curr_f:
                        continue
                    if load[new_f] + demands[i] > capacities[new_f]:
                        continue
                    
                    delta = assign_cost[i, new_f] - assign_cost[i, curr_f]
                    
                    # Open cost logic
                    if curr_load == demands[i]:
                        if np.sum(assignment == curr_f) == 1:
                            delta -= open_cost[curr_f]
                    
                    if load[new_f] == 0:
                        delta += open_cost[new_f]
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_f = new_f
                
                if best_f != curr_f:
                    load[curr_f] -= demands[i]
                    load[best_f] += demands[i]
                    assignment[i] = best_f
                    improved = True
        
        return assignment
    
    def lagrangian_relaxation():
        """
        Solve via Lagrangian Relaxation:
        - Dualize capacity constraints
        - Iteratively update multipliers via subgradient method
        - At each iteration, solve decomposed customer subproblems
        """
        # Multipliers for capacity constraints (one per facility)
        lambda_ = np.zeros(n_facilities)
        
        # Track best feasible solution
        best_assign = None
        best_cost = np.inf
        
        # Subgradient step size parameters
        step_scale = 1.0
        max_iter = 500
        
        for it in range(max_iter):
            if time.time() > deadline - 15:
                break
            
            # Solve decomposed subproblem:
            # For each customer i, choose facility f minimizing:
            #   assign_cost[i,f] + lambda_[f] * demands[i]
            # But also consider opening costs via a heuristic adjustment
            # 
            # Actually, the Lagrangian relaxation of capacity constraints gives:
            # L(lambda) = min_{x} sum_{i,f} (assign_cost[i,f] + lambda[f]*demands[i]) * x[i,f]
            #           + sum_f (open_cost[f]*y[f] - lambda[f]*capacities[f])
            #           subject to sum_f x[i,f] = 1 for all i
            #
            # This decomposes into independent customer problems.
            # For each customer i: argmin_f (assign_cost[i,f] + lambda[f]*demands[i])
            #
            # The dual function is:
            # g(lambda) = sum_i min_f (assign_cost[i,f] + lambda[f]*demands[i])
            #            + sum_f (open_cost[f]*y_f*(lambda) - lambda[f]*capacities[f])
            #
            # where y_f*(lambda) = 1 if facility f is used by at least one customer in the primal solution
            
            # Solve primal subproblem
            adjusted_cost = assign_cost + np.outer(demands, lambda_)
            assignment = np.argmin(adjusted_cost, axis=1).astype(int)
            
            # Determine which facilities are open
            open_mask = np.zeros(n_facilities, dtype=bool)
            for f in range(n_facilities):
                if np.any(assignment == f):
                    open_mask[f] = True
            
            # Calculate dual function value
            dual_value = np.sum(adjusted_cost[np.arange(n_customers), assignment])
            dual_value += np.sum(open_cost[open_mask]) - np.sum(lambda_ * capacities)
            
            # Check if this assignment is feasible (capacity constraints)
            load = np.zeros(n_facilities)
            for i in range(n_customers):
                load[assignment[i]] += demands[i]
            
            feasible = np.all(load <= capacities + 1e-6)
            
            if feasible:
                cost = calc_cost(assignment)
                if cost < best_cost:
                    best_cost = cost
                    best_assign = assignment.copy()
                    # If we found a feasible solution, we can try to improve it
                    # But first continue with subgradient updates
            
            # Compute subgradient of g(lambda)
            # sg[f] = (sum_i demands[i]*x[i,f]) - capacities[f]*y_f*(lambda)
            # where x[i,f] = 1 if customer i is assigned to facility f
            sg = np.zeros(n_facilities)
            for f in range(n_facilities):
                sg[f] = load[f] - capacities[f] * (1 if open_mask[f] else 0)
            
            # Step size (Polyak-style or diminishing)
            # Use diminishing step size
            step_size = step_scale / (1.0 + 0.1 * it)
            
            # Update multipliers (project onto lambda >= 0)
            lambda_ = np.maximum(0, lambda_ + step_size * sg)
            
            # Occasionally try to repair infeasible solutions
            if not feasible and it % 10 == 0:
                # Try to fix by moving customers from overloaded facilities
                for f in range(n_facilities):
                    if load[f] > capacities[f]:
                        # Move customers from f to other facilities
                        customers_at_f = np.where(assignment == f)[0]
                        for i in customers_at_f:
                            # Find best feasible alternative
                            best_alt = -1
                            best_alt_cost = np.inf
                            for alt_f in range(n_facilities):
                                if alt_f == f:
                                    continue
                                if load[alt_f] + demands[i] <= capacities[alt_f]:
                                    alt_cost = assign_cost[i, alt_f]
                                    if alt_cost < best_alt_cost:
                                        best_alt_cost = alt_cost
                                        best_alt = alt_f
                            if best_alt >= 0:
                                load[f] -= demands[i]
                                load[best_alt] += demands[i]
                                assignment[i] = best_alt
                                # Check if f is now empty
                                if load[f] == 0:
                                    open_mask[f] = False
            
            # Check if we should stop
            if it > 100 and it % 20 == 0:
                # Try local search on best feasible solution
                if best_assign is not None:
                    improved = local_search(best_assign, time_budget=5.0)
                    improved_cost = calc_cost(improved)
                    if improved_cost < best_cost:
                        best_cost = improved_cost
                        best_assign = improved
        
        return best_assign
    
    # Try Lagrangian Relaxation first
    lr_result = None
    try:
        lr_result = lagrangian_relaxation()
    except Exception as e:
        pass
    
    if lr_result is not None:
        # Polish with local search
        lr_result = local_search(lr_result, time_budget=min(20.0, deadline - time.time() - 5))
        lr_cost = calc_cost(lr_result)
        
        # Try MILP if we have time
        if time.time() < deadline - 30:
            try:
                n_x = n_customers * n_facilities
                n_vars = n_x + n_facilities
                
                c = np.zeros(n_vars)
                for f in range(n_facilities):
                    c[n_x + f] = open_cost[f]
                for i in range(n_customers):
                    for f in range(n_facilities):
                        c[i * n_facilities + f] = assign_cost[i, f]
                
                integrality = np.ones(n_vars)
                lb_vars = np.zeros(n_vars)
                ub_vars = np.ones(n_vars)
                
                A_rows = []
                lb_rows = []
                ub_rows = []
                
                for i in range(n_customers):
                    row = np.zeros(n_vars)
                    for f in range(n_facilities):
                        row[i * n_facilities + f] = 1.0
                    A_rows.append(row)
                    lb_rows.append(1.0)
                    ub_rows.append(1.0)
                
                for f in range(n_facilities):
                    row = np.zeros(n_vars)
                    for i in range(n_customers):
                        row[i * n_facilities + f] = demands[i]
                    row[n_x + f] = -capacities[f]
                    A_rows.append(row)
                    lb_rows.append(-np.inf)
                    ub_rows.append(0.0)
                
                A = np.array(A_rows)
                lb_c = np.array(lb_rows)
                ub_c = np.array(ub_rows)
                
                constraint = LinearConstraint(A, lb_c, ub_c)
                bounds = Bounds(lb_vars, ub_vars)
                
                result = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
                
                if result.success and result.x is not None:
                    x_flat = result.x[:n_x]
                    x_matrix = x_flat.reshape(n_customers, n_facilities)
                    milp_assign = np.argmax(x_matrix, axis=1).astype(int)
                    
                    load = np.zeros(n_facilities)
                    for i in range(n_customers):
                        load[milp_assign[i]] += demands[i]
                    
                    if np.all(load <= capacities + 1e-6):
                        milp_assign = local_search(milp_assign, time_budget=min(10.0, deadline - time.time() - 2))
                        milp_cost = calc_cost(milp_assign)
                        
                        if milp_cost < lr_cost:
                            return milp_assign
            except Exception:
                pass
        
        return lr_result
    
    # Fallback: Greedy + Local Search
    def greedy_seed(order_type='desc'):
        res = np.full(n_customers, -1, dtype=int)
        load = np.zeros(n_facilities)
        if order_type == 'desc':
            order = np.argsort(-demands)
        else:
            order = np.argsort(demands)
        
        for i in order:
            fac_order = np.argsort(assign_cost[i])
            feasible = [f for f in fac_order if load[f] + demands[i] <= capacities[f]]
            if feasible:
                f_best = min(feasible, key=lambda f: assign_cost[i, f])
                res[i] = f_best
                load[f_best] += demands[i]
            else:
                f_best = np.argmax(capacities - load)
                res[i] = f_best
                load[f_best] += demands[i]
        
        return res
    
    best_assign = greedy_seed('desc')
    best_assign = local_search(best_assign, time_budget=min(15.0, deadline - time.time() - 5))
    best_cost = calc_cost(best_assign)
    
    if time.time() < deadline - 10:
        alt_assign = greedy_seed('asc')
        alt_assign = local_search(alt_assign, time_budget=min(10.0, deadline - time.time() - 3))
        alt_cost = calc_cost(alt_assign)
        if alt_cost < best_cost:
            best_assign = alt_assign
    
    return best_assign
