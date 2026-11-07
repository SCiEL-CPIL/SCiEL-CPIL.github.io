"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment


def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # We need to assign each customer to a facility such that:
    # 1. Each customer is assigned to exactly one facility
    # 2. The total demand assigned to each facility does not exceed its capacity
    # 3. Minimize sum of open_cost for opened facilities + sum of assign_cost
    
    # This is a capacitated facility location problem, which is NP-hard.
    # We'll use a heuristic approach with local search.
    
    # Strategy: 
    # 1. Start with a feasible assignment using a greedy approach
    # 2. Improve via local search: try reassigning customers to different facilities
    #    to reduce total cost while maintaining feasibility
    
    # First, let's create an initial feasible assignment
    # Sort customers by demand descending (hardest to place first)
    customer_order = np.argsort(-demands)
    
    result = np.full(n_customers, -1, dtype=int)
    load = np.zeros(n_facilities)
    
    # Greedy: assign each customer to the cheapest feasible facility
    for i in customer_order:
        # Get facilities sorted by assignment cost for this customer
        fac_order = np.argsort(assign_cost[i])
        assigned = False
        for f in fac_order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                assigned = True
                break
        if not assigned:
            # This shouldn't happen if total capacity >= total demand
            # But if it does, assign to facility with most remaining capacity
            remaining = capacities - load
            f = np.argmax(remaining)
            result[i] = f
            load[f] += demands[i]
    
    # Now compute initial cost
    def compute_cost(assignment, loads):
        # Sum of assignment costs
        assign_sum = 0.0
        for i in range(n_customers):
            assign_sum += assign_cost[i, assignment[i]]
        
        # Sum of open costs for facilities with at least one customer
        open_sum = 0.0
        used_facilities = set(assignment.tolist())
        for f in used_facilities:
            open_sum += open_cost[f]
        
        return assign_sum + open_sum
    
    current_cost = compute_cost(result, load)
    
    # Local search: try improving by reassigning individual customers
    # or swapping customers between facilities
    max_iterations = 5000
    improvement_threshold = 1e-9
    
    for iteration in range(max_iterations):
        improved = False
        
        # Try reassigning each customer to a different facility
        for i in range(n_customers):
            current_f = result[i]
            if current_f < 0:
                continue
            
            # Calculate cost if we move customer i to each other facility
            for f in range(n_facilities):
                if f == current_f:
                    continue
                
                # Check capacity constraint
                if load[f] + demands[i] > capacities[f]:
                    continue
                
                # Calculate new cost
                # Remove customer i from current_f
                old_load_current = load[current_f]
                load[current_f] = old_load_current - demands[i]
                
                # Add customer i to f
                load[f] += demands[i]
                
                # Compute cost difference
                # Assignment cost change
                assign_delta = assign_cost[i, f] - assign_cost[i, current_f]
                
                # Open cost change
                # Check if current_f becomes empty
                open_delta = 0.0
                if old_load_current - demands[i] == 0:
                    # current_f is now empty, save open cost
                    open_delta -= open_cost[current_f]
                
                # Check if f was empty before
                if old_load_current - demands[i] == load[current_f]:  # This is always true, let me rethink
                    pass
                
                # Actually, let's just recompute the open cost part properly
                # Old open facilities
                old_used = set(result.tolist())
                # But we're changing result, so let's track it differently
                
                # Let me recompute more carefully
                # Temporarily update result
                old_result_i = result[i]
                result[i] = f
                
                new_cost = compute_cost(result, load)
                result[i] = old_result_i
                
                # Revert load
                load[f] -= demands[i]
                load[current_f] = old_load_current
                
                if new_cost < current_cost - improvement_threshold:
                    # Apply the move
                    load[current_f] -= demands[i]
                    load[f] += demands[i]
                    result[i] = f
                    current_cost = new_cost
                    improved = True
                    break  # Restart after improvement
        
        if not improved:
            break
    
    # Final local search: try swapping pairs of customers between facilities
    for iteration in range(1000):
        improved = False
        
        for i in range(n_customers):
            for j in range(i + 1, n_customers):
                fi = result[i]
                fj = result[j]
                
                if fi == fj:
                    continue
                
                # Try swapping: i -> fj, j -> fi
                # Check capacity constraints
                if load[fj] - demands[j] + demands[i] > capacities[fj]:
                    continue
                if load[fi] - demands[i] + demands[j] > capacities[fi]:
                    continue
                
                # Compute cost difference
                old_assign = assign_cost[i, fi] + assign_cost[j, fj]
                new_assign = assign_cost[i, fj] + assign_cost[j, fi]
                assign_delta = new_assign - old_assign
                
                # Open cost change
                # After swap, check if fi or fj become empty
                old_load_fi = load[fi]
                old_load_fj = load[fj]
                
                new_load_fi = old_load_fi - demands[i] + demands[j]
                new_load_fj = old_load_fj - demands[j] + demands[i]
                
                open_delta = 0.0
                if old_load_fi == demands[i] and new_load_fi > 0:
                    # fi was only serving i, now serves j, still open - no change
                    pass
                elif old_load_fi == demands[i] and new_load_fi == 0:
                    # fi becomes empty
                    open_delta -= open_cost[fi]
                elif old_load_fi > demands[i] and new_load_fi == 0:
                    # fi becomes empty (was serving i and others, but new_load_fi = old - demands[i] + demands[j] = 0)
                    open_delta -= open_cost[fi]
                
                if old_load_fj == demands[j] and new_load_fj == 0:
                    open_delta -= open_cost[fj]
                elif old_load_fj > demands[j] and new_load_fj == 0:
                    open_delta -= open_cost[fj]
                
                # Also check if facilities that were empty become non-empty
                # But since we're swapping between two non-empty facilities, this shouldn't happen
                
                if assign_delta + open_delta < -improvement_threshold:
                    # Apply the swap
                    load[fi] = new_load_fi
                    load[fj] = new_load_fj
                    result[i] = fj
                    result[j] = fi
                    current_cost += assign_delta + open_delta
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return result
# EVOLVE-BLOCK-END
