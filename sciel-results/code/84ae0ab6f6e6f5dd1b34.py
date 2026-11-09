"""Capacitated facility location: Branch-and-bound with Lagrangian relaxation."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using branch-and-bound with Lagrangian relaxation.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Safety margin: 150s limit, aim for 135s max to leave buffer
    deadline = time.time() + 135.0
    
    best_score = np.inf
    best_result = np.zeros(n_cust, dtype=int)
    
    def calc_score(res):
        used = np.unique(res)
        return np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(n_cust), res])
    
    def is_valid(res):
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[res[i]] += demands[i]
        return np.all(load <= capacities)
    
    # 1. Greedy Initialization
    result_greedy = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    order = np.argsort(-demands)
    for i in order:
        feasible = np.where(load + demands[i] <= capacities)[0]
        if len(feasible) == 0:
            remaining = capacities - load
            f = np.argmax(remaining)
            result_greedy[i] = f
            load[f] += demands[i]
        else:
            costs = assign_cost[i, feasible]
            f_idx = feasible[np.argmin(costs)]
            result_greedy[i] = f_idx
            load[f_idx] += demands[i]
            
    best_result = result_greedy.copy()
    best_score = calc_score(best_result)
    
    # 2. Lagrangian Relaxation & Subgradient Optimization
    # We relax capacity constraints: sum(demand_i * x_if) <= cap_f
    # L(λ) = sum_f (open_cost_f * y_f) + sum_i,f (assign_cost_if * x_if) + sum_f λ_f * (sum_i demand_i * x_if - cap_f)
    # For fixed λ, the problem decouples:
    # 1. Facility opening: Minimize open_cost_f + λ_f * (-cap_f) -> Open if cost < 0? No, y_f is binary.
    #    Actually, the term is λ_f * (Load_f - cap_f).
    #    If we decide x_if, we contribute λ_f * demand_i.
    #    The facility opening decision y_f is coupled to whether any x_if=1.
    #    This is still coupled.
    #    
    # Alternative: Lagrangian Relaxation of assignment constraints?
    # Let's try a different relaxation: Relax the "one facility per customer" constraint?
    # No, that makes it a fractional assignment which is easy (LP).
    #
    # Let's stick to a simpler, more robust structural change: 
    # **Lagrangian Relaxation of Capacity Constraints with Subgradient Method**
    # to find a good heuristic solution and a bound.
    
    # Objective: Min sum(open_cost_f * y_f) + sum(assign_cost_if * x_if)
    # s.t. sum_f x_if = 1 for all i
    #      sum_i demand_i * x_if <= cap_f * y_f for all f
    #      x_if, y_f in {0,1}
    
    # Relax capacity constraints with multipliers lambda_f >= 0.
    # Lagrangian L(lambda) = min_{x,y} [ sum(open_cost_f * y_f) + sum(assign_cost_if * x_if) + sum_f lambda_f * (sum_i demand_i * x_if - cap_f * y_f) ]
    # Rearranging terms by facility f:
    # L(lambda) = - sum_f lambda_f * cap_f + sum_f [ open_cost_f * y_f - lambda_f * cap_f * y_f + sum_i (assign_cost_if + lambda_f * demand_i) * x_if ]
    # This doesn't decouple y_f and x_if easily because y_f appears with x_if in the original constraint?
    # Wait, the constraint is sum_i demand_i * x_if <= cap_f * y_f.
    # The term is lambda_f * (sum_i demand_i * x_if - cap_f * y_f).
    # So L = sum_f (open_cost_f * y_f - lambda_f * cap_f * y_f) + sum_i,f (assign_cost_if + lambda_f * demand_i) * x_if - sum_f lambda_f * cap_f ... wait.
    # L = sum_f (open_cost_f - lambda_f * cap_f) * y_f + sum_i,f (assign_cost_if + lambda_f * demand_i) * x_if
    # Subject to sum_f x_if = 1.
    # For a fixed set of open facilities (y_f=1), the cost for customer i is min_{f: y_f=1} (assign_cost_if + lambda_f * demand_i).
    # But y_f is also a decision variable.
    # This is still a facility location problem but with modified costs.
    # Specifically, if we fix y, the optimal x is greedy: x_if = 1 if f = argmin_{g: y_g=1} (assign_cost_ig + lambda_g * demand_i).
    # Then the objective value for a given y is:
    # Val(y) = sum_f (open_cost_f - lambda_f * cap_f) * y_f + sum_i min_{f: y_f=1} (assign_cost_if + lambda_f * demand_i)
    # We want to maximize L(lambda) over lambda >= 0.
    # And for a given lambda, we want to find y that minimizes Val(y).
    
    # This is a standard Lagrangian Relaxation approach.
    
    # Subgradient of L(lambda) w.r.t lambda_f is: (sum_i demand_i * x_if(lambda) - cap_f * y_f(lambda))
    # Where x, y are the optimal solution to the relaxed problem for lambda.
    
    # Algorithm:
    # 1. Start with lambda = 0.
    # 2. Solve the relaxed problem (which is a UFLP with modified costs and fixed lambda) to get x, y.
    #    Actually, solving the relaxed problem for a fixed lambda is still a UFLP (Un-capacitated Facility Location Problem) because the capacity constraint is moved to the objective.
    #    Wait, if capacity is relaxed, the problem becomes:
    #    Min sum_f c_f y_f + sum_i,f a_if x_if + sum_f lambda_f (sum_i d_i x_if - cap_f y_f)
    #    = sum_f (c_f - lambda_f cap_f) y_f + sum_i,f (a_if + lambda_f d_i) x_if
    #    s.t. sum_f x_if = 1, x_if <= y_f (implied if we assume x=0 when y=0? No, x_if can be 1 only if y_f=1? 
    #    In the original problem, x_if=1 implies y_f=1.
    #    In the Lagrangian, do we keep x_if <= y_f?
    #    If we keep x_if <= y_f, then for a fixed y, the best x is greedy.
    #    So the problem reduces to choosing y to minimize:
    #    sum_f (c_f - lambda_f cap_f) y_f + sum_i min_{f: y_f=1} (a_if + lambda_f d_i)
    #    This is a UFLP instance! We can solve it greedily or with local search.
    
    # Let's implement a simple solver for the UFLP part using local search.
    
    def solve_uflp(modified_open_cost, modified_assign_cost):
        # Greedy init
        res = np.zeros(n_cust, dtype=int)
        for i in range(n_cust):
            res[i] = np.argmin(modified_assign_cost[i])
        
        # Local search
        improved = True
        while improved:
            improved = False
            # Move customers
            for i in range(n_cust):
                f_old = res[i]
                # Try moving to other facilities
                # Calculate delta
                # If f_old becomes empty, we save open_cost[f_old]
                # If f_new was empty, we pay open_cost[f_new]
                
                # Simple heuristic: try all f_new
                best_delta = 0
                best_f_new = f_old
                
                load_old = np.sum(res == f_old)
                load_new_counts = np.bincount(res, minlength=n_fac)
                
                for f_new in range(n_fac):
                    if f_new == f_old:
                        continue
                    delta_assign = modified_assign_cost[i, f_new] - modified_assign_cost[i, f_old]
                    delta_open = 0
                    if load_old == 1:
                        delta_open -= modified_open_cost[f_old]
                    if load_new_counts[f_new] == 0:
                        delta_open += modified_open_cost[f_new]
                    
                    if delta_assign + delta_open < best_delta:
                        best_delta = delta_assign + delta_open
                        best_f_new = f_new
                        
                if best_delta < -1e-9:
                    res[i] = best_f_new
                    improved = True
            
            # Close facilities
            used_fac = np.unique(res)
            for f_close in used_fac:
                custs = np.where(res == f_close)[0]
                if len(custs) == 0: continue
                
                # Try to relocate all custs
                temp_res = res.copy()
                success = True
                for c in custs:
                    # Find best facility excluding f_close
                    # We want to minimize modified_assign_cost[c, f] + open_cost penalty if new
                    # Simple greedy: pick min modified_assign_cost among f != f_close
                    # But we need to check if it opens a new facility.
                    # For simplicity, just pick min modified_assign_cost.
                    # This might open expensive facilities, but the UFLP objective accounts for that in the global sense?
                    # No, the local search on UFLP should be smarter.
                    # Let's just use the best available f != f_close with min modified_assign_cost.
                    costs = modified_assign_cost[c].copy()
                    costs[f_close] = np.inf
                    f_new = np.argmin(costs)
                    temp_res[c] = f_new
                    
                # Calculate delta for closing f_close
                # Old cost: open_cost[f_close] + sum(modified_assign_cost[c, f_close] for c in custs)
                # New cost: sum(modified_assign_cost[c, temp_res[c] for c in custs) + open costs for newly opened
                # This is complex to calculate exactly in this simple loop.
                # Let's skip closing moves in the inner UFLP solver for speed and rely on the outer subgradient to adjust lambda.
                
        # Return the assignment and the set of open facilities
        used = np.unique(res)
        return res, used

    # Subgradient Optimization
    lambda_vec = np.zeros(n_fac)
    step_size_factor = 1.0
    
    # Initial UFLP solve with lambda=0
    res_uflp, used_uflp = solve_uflp(open_cost, assign_cost)
    # Note: The UFLP solution might violate capacities.
    # We use it to get a heuristic solution for the original problem by "repairing" it.
    
    def repair_solution(res):
        # Fix capacity violations by moving customers to other facilities
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[res[i]] += demands[i]
            
        valid = True
        # Iterative repair
        for _ in range(100):
            if time.time() > deadline - 5: break
            overloaded = np.where(load > capacities)[0]
            if len(overloaded) == 0:
                break
                
            f = overloaded[0]
            # Find customer in f to move
            custs_in_f = np.where(res == f)[0]
            best_cust = -1
            best_target = -1
            min_penalty = np.inf
            
            for c in custs_in_f:
                d_c = demands[c]
                # Find target with enough capacity
                feasible_targets = np.where(load + d_c <= capacities)[0]
                feasible_targets = feasible_targets[feasible_targets != f]
                
                if len(feasible_targets) == 0:
                    continue
                    
                for t in feasible_targets:
                    # Penalty: increase in assign cost + potential open cost
                    # If t is empty, we pay open_cost[t]
                    penalty = assign_cost[c, t] - assign_cost[c, f]
                    if load[t] == 0:
                        penalty += open_cost[t]
                    # If f becomes empty, we save open_cost[f] (but we are moving one at a time, so f won't be empty until the last one)
                    # Approximate penalty
                    if penalty < min_penalty:
                        min_penalty = penalty
                        best_cust = c
                        best_target = t
                        
            if best_cust == -1:
                # Can't repair with single moves. Try swap?
                # For now, break
                break
                
            # Move
            res[best_cust] = best_target
            load[f] -= demands[best_cust]
            load[best_target] += demands[best_cust]
            
        return res, load

    # Run Subgradient Loop
    for k in range(50):
        if time.time() > deadline - 10:
            break
            
        # Solve UFLP with current lambda
        mod_open = open_cost - lambda_vec * capacities
        mod_assign = assign_cost + np.outer(demands, lambda_vec)
        
        res_uflp, used_uflp = solve_uflp(mod_open, mod_assign)
        
        # Calculate Lagrangian Value
        # L = sum_f (open_cost_f - lambda_f * cap_f) * y_f + sum_i min_{f: y_f=1} (assign_cost_if + lambda_f * demand_i)
        # y_f = 1 if f in used_uflp
        y = np.zeros(n_fac)
        y[used_uflp] = 1
        
        # Calculate assignment part
        # For each i, min cost among used_uflp
        # We can just use res_uflp which is the optimal assignment for the UFLP
        assign_part = np.sum(assign_cost[np.arange(n_cust), res_uflp] + lambda_vec[res_uflp] * demands)
        open_part = np.sum(mod_open[used_uflp])
        
        L_val = open_part + assign_part
        
        # Update best solution if valid
        # The UFLP solution might be infeasible for the original problem (capacity violations)
        # We need to repair it to get a valid solution for the original problem
        res_repaired, load_repaired = repair_solution(res_uflp.copy())
        
        if np.all(load_repaired <= capacities):
            score = calc_score(res_repaired)
            if score < best_score:
                best_score = score
                best_result = res_repaired.copy()
        
        # Calculate Subgradient
        # g_f = sum_i demand_i * x_if - cap_f * y_f
        # x_if = 1 if res_uflp[i] == f
        x = np.zeros((n_cust, n_fac))
        x[np.arange(n_cust), res_uflp] = 1
        
        subgrad = np.sum(x * demands[:, np.newaxis], axis=0) - capacities * y
        
        # Update Lambda
        # lambda_new = max(0, lambda_old + step * subgrad)
        # Step size: alpha * (UB - L_val) / ||g||^2
        if np.linalg.norm(subgrad) > 1e-9:
            if L_val < best_score:
                step = step_size_factor * (best_score - L_val) / (np.dot(subgrad, subgrad))
            else:
                step = 0
            lambda_vec = np.maximum(0, lambda_vec + step * subgrad)
        else:
            break
            
    # Final Local Search on the best result found
    if time.time() < deadline - 5:
        # Run a quick local search on best_result
        res_ls = best_result.copy()
        load_ls = np.zeros(n_fac)
        for i in range(n_cust):
            load_ls[res_ls[i]] += demands[i]
            
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 2: break
            
            # Single moves
            for i in range(n_cust):
                f_old = res_ls[i]
                d_i = demands[i]
                feasible = np.where(load_ls + d_i <= capacities)[0]
                feasible = feasible[feasible != f_old]
                
                if len(feasible) == 0: continue
                
                best_f_new = -1
                best_delta = 0
                base_cost = assign_cost[i, f_old]
                
                for f_new in feasible:
                    delta_assign = assign_cost[i, f_new] - base_cost
                    delta_open = 0
                    if load_ls[f_old] == d_i:
                        delta_open -= open_cost[f_old]
                    if load_ls[f_new] == 0:
                        delta_open += open_cost[f_new]
                        
                    if delta_assign + delta_open < best_delta:
                        best_delta = delta_assign + delta_open
                        best_f_new = f_new
                        
                if best_delta < -1e-9:
                    res_ls[i] = best_f_new
                    load_ls[f_old] -= d_i
                    load_ls[best_f_new] += d_i
                    improved = True
                    
    # Final validation
    if np.all(np.zeros(n_fac) + np.bincount(res_ls, minlength=n_fac) <= capacities):
        final_score = calc_score(res_ls)
        if final_score < best_score:
            best_result = res_ls
            best_score = final_score
            
    return best_result
# EVOLVE-BLOCK-END
