import numpy as np

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Initialize Lagrange multipliers
    lambdas = np.zeros(n_facilities)
    
    best_solution = None
    best_cost = float('inf')
    
    # Number of Lagrangian iterations
    num_iters = 150
    
    # Helper to calculate cost of a feasible assignment
    def calc_cost(assignment):
        cost = 0.0
        # Open costs
        for f in range(n_facilities):
            if np.any(assignment == f):
                cost += open_cost[f]
        # Assignment costs
        cost += np.sum(assign_cost[np.arange(n_customers), assignment])
        return cost

    # Helper to construct a feasible solution using greedy with Lagrangian costs
    def construct_feasible(lmb):
        # Cost matrix including Lagrangian penalty
        # Cost for customer i to facility f: assign_cost[i, f] + lmb[f] * demands[i]
        # We want to minimize this, subject to capacity.
        
        # Sort customers by demand descending (largest first)
        order = np.argsort(-demands)
        
        assignment = np.full(n_customers, -1, dtype=int)
        load = np.zeros(n_facilities)
        
        for i in order:
            # Calculate effective costs for this customer
            eff_costs = assign_cost[i] + lmb * demands[i]
            
            # Find feasible facilities (sorted by effective cost)
            # We need to check capacity for each candidate
            # To be efficient, sort facilities by eff_cost
            sorted_f = np.argsort(eff_costs)
            
            assigned = False
            for f in sorted_f:
                if load[f] + demands[i] <= capacities[f]:
                    assignment[i] = f
                    load[f] += demands[i]
                    assigned = True
                    break
            
            if not assigned:
                # Fallback: find any facility with space, pick min raw cost
                for f in np.argsort(assign_cost[i]):
                    if load[f] + demands[i] <= capacities[f]:
                        assignment[i] = f
                        load[f] += demands[i]
                        assigned = True
                        break
                if not assigned:
                    # Should not happen if total capacity >= total demand
                    # Assign to facility with most remaining capacity
                    remaining = capacities - load
                    f_max = np.argmax(remaining)
                    assignment[i] = f_max
                    load[f_max] += demands[i]
                    
        return assignment

    # Helper for local search (1-move)
    def local_search(assignment, max_moves=100):
        load = np.zeros(n_facilities)
        for i in range(n_customers):
            load[assignment[i]] += demands[i]
            
        current_cost = calc_cost(assignment)
        improved = True
        moves_made = 0
        
        while improved and moves_made < max_moves:
            improved = False
            for i in range(n_customers):
                old_f = assignment[i]
                # Try moving i to another facility
                # Sort other facilities by assign_cost[i, f]
                other_f = np.argsort(assign_cost[i])
                
                for new_f in other_f:
                    if new_f == old_f:
                        continue
                    if load[new_f] + demands[i] > capacities[new_f]:
                        continue
                    
                    # Calculate delta
                    # Old contribution: assign_cost[i, old_f] + (open_cost[old_f] if load[old_f]==demands[i] else 0)
                    # New contribution: assign_cost[i, new_f] + (open_cost[new_f] if load[new_f]==0 else 0)
                    
                    # Note: open_cost[old_f] is saved if load[old_f] becomes 0
                    # open_cost[new_f] is added if load[new_f] was 0
                    
                    old_open = open_cost[old_f] if load[old_f] == demands[i] else 0.0
                    new_open = open_cost[new_f] if load[new_f] == 0.0 else 0.0
                    
                    delta = (assign_cost[i, new_f] + new_open) - (assign_cost[i, old_f] + old_open)
                    
                    if delta < -1e-9:
                        # Apply move
                        load[old_f] -= demands[i]
                        load[new_f] += demands[i]
                        assignment[i] = new_f
                        current_cost += delta
                        improved = True
                        moves_made += 1
                        break
                        
        return assignment, current_cost

    for it in range(num_iters):
        # 1. Solve relaxed problem to get gradient
        # Cost matrix: assign_cost + outer(demands, lambdas)
        relaxed_costs = assign_cost + np.outer(demands, lambdas)
        relaxed_assignment = np.argmin(relaxed_costs, axis=1)
        
        # Calculate load of relaxed assignment
        relaxed_load = np.zeros(n_facilities)
        for i in range(n_customers):
            relaxed_load[relaxed_assignment[i]] += demands[i]
            
        # 2. Update Lagrange multipliers (Subgradient method)
        # Gradient g = relaxed_load - capacities
        g = relaxed_load - capacities
        
        # Step size: start with 1.0, decrease slowly
        # Simple step size rule: step = alpha / ||g||^2
        # Or constant step. Let's use constant 0.5 for stability
        step = 0.5
        # Ensure lambdas don't become negative
        lambdas = np.maximum(0, lambdas + step * g)
        
        # 3. Construct feasible solution
        feasible_sol = construct_feasible(lambdas)
        
        # 4. Polish with local search
        polished_sol, sol_cost = local_search(feasible_sol, max_moves=20)
        
        if sol_cost < best_cost:
            best_cost = sol_cost
            best_solution = polished_sol.copy()

    # Final verification and return
    if best_solution is None:
        # Fallback
        best_solution = np.zeros(n_customers, dtype=int)
        
    # Check validity
    load_check = np.zeros(n_facilities)
    for i in range(n_customers):
        f = best_solution[i]
        if f < 0 or f >= n_facilities:
            return np.zeros(n_customers, dtype=int)
        load_check[f] += demands[i]
        
    for f in range(n_facilities):
        if load_check[f] > capacities[f]:
            # Re-run simple greedy if invalid (should not happen)
            result = np.full(n_customers, -1, dtype=int)
            load = np.zeros(n_facilities)
            for i in np.argsort(-demands):
                for f in np.argsort(assign_cost[i]):
                    if load[f] + demands[i] <= capacities[f]:
                        result[i] = f
                        load[f] += demands[i]
                        break
            if -1 in result:
                result = np.zeros(n_customers, dtype=int)
            return result
            
    return best_solution
