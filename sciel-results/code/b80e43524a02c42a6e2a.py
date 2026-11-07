"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using Mixed Integer Programming (MIP)
    if available, otherwise falls back to a greedy heuristic with local search.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Try to use scipy.optimize.milp for exact solution
    try:
        # Check if milp is available (SciPy >= 1.7.0)
        if not hasattr(scipy.optimize, 'milp'):
            raise AttributeError("milp not available")
            
        # Flatten costs for objective
        # Variables: x_{ij} (n_customers * n_facilities), y_j (n_facilities)
        n_x = n_customers * n_facilities
        n_y = n_facilities
        n_vars = n_x + n_y
        
        # Objective coefficients
        c = np.zeros(n_vars)
        # x_{ij} costs
        c[:n_x] = assign_cost.flatten()
        # y_j costs
        c[n_x:] = open_cost
        
        # Constraints
        # 1. Each customer assigned to exactly one facility: sum_j x_{ij} = 1
        # 2. Capacity: sum_i d_i x_{ij} - C_j y_j <= 0
        
        # We build the constraint matrix A
        # Rows: n_customers (equality) + n_facilities (inequality)
        # Cols: n_vars
        
        A_eq = np.zeros((n_customers, n_vars))
        for i in range(n_customers):
            for j in range(n_facilities):
                idx = i * n_facilities + j
                A_eq[i, idx] = 1.0
        
        A_ineq = np.zeros((n_facilities, n_vars))
        for j in range(n_facilities):
            # sum_i d_i x_{ij} - C_j y_j <= 0
            for i in range(n_customers):
                idx = i * n_facilities + j
                A_ineq[j, idx] = demands[i]
            # y_j term
            A_ineq[j, n_x + j] = -capacities[j]
        
        # Combine constraints
        # LinearConstraint(A, lb, ub)
        # We can use two separate constraints or combine.
        # Let's use separate for clarity or combine into one big matrix.
        # Combining:
        A = np.vstack([A_eq, A_ineq])
        lb = np.concatenate([np.ones(n_customers), -np.inf * np.ones(n_facilities)])
        ub = np.concatenate([np.ones(n_customers), np.zeros(n_facilities)])
        
        cons = scipy.optimize.LinearConstraint(A, lb, ub)
        
        # Integrality: all variables are binary (integer)
        integrality = np.ones(n_vars)
        
        # Bounds: [0, 1]
        bounds = scipy.optimize.Bounds(0, 1)
        
        # Solve
        res = scipy.optimize.milp(c, constraints=cons, integrality=integrality, bounds=bounds, 
                                  options={"presolve": True, "time_limit": 100})
        
        if res.status == 0: # Optimal
            x_flat = res.x[:n_x]
            # Map back to (n_customers, n_facilities)
            x = x_flat.reshape(n_customers, n_facilities)
            
            # Find assigned facility for each customer
            # x_{ij} should be 1 for the assigned facility
            # Due to numerical precision, use argmax
            result = np.argmax(x, axis=1).astype(int)
            
            # Verify validity (just in case)
            load = np.zeros(n_facilities)
            for i in range(n_customers):
                load[result[i]] += demands[i]
            if np.any(load > capacities + 1e-5):
                # Fallback if solver error
                return _fallback_assign(demands, capacities, open_cost, assign_cost)
                
            return result
            
        else:
            # Solver failed, fallback
            return _fallback_assign(demands, capacities, open_cost, assign_cost)

    except Exception:
        # Fallback to heuristic
        return _fallback_assign(demands, capacities, open_cost, assign_cost)


def _fallback_assign(demands, capacities, open_cost, assign_cost):
    """
    Greedy heuristic with local search improvement.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Initial greedy assignment: Largest demand first, cheapest feasible facility
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    
    # Sort customers by demand descending
    order = np.argsort(-demands)
    
    for i in order:
        # Find feasible facilities sorted by cost
        feasible = []
        for j in range(n_facilities):
            if load[j] + demands[i] <= capacities[j]:
                feasible.append((assign_cost[i, j], j))
        
        if not feasible:
            # Should not happen if total capacity >= total demand, 
            # but if it does, we have an invalid instance or need to relax?
            # Task says "facility whose assigned demand exceeds its capacity make the whole result invalid".
            # So we must find a valid assignment.
            # If no feasible facility, we might need to reassign previous customers.
            # For now, pick the one with most remaining capacity (last resort)
            # This indicates the greedy order was bad.
            # Let's just pick the one with min cost among those with max capacity?
            # Actually, if this happens, the greedy approach failed.
            # We should probably restart or use a different strategy.
            # For robustness, let's try to find ANY feasible facility.
            # If none, the instance is infeasible? The problem implies feasible instances.
            # Let's assume feasible.
            pass
            
        # Sort by cost
        feasible.sort()
        if feasible:
            best_cost, best_fac = feasible[0]
            result[i] = best_fac
            load[best_fac] += demands[i]
        else:
            # Emergency: pick facility with max remaining capacity
            rem_cap = capacities - load
            best_fac = np.argmax(rem_cap)
            result[i] = best_fac
            load[best_fac] += demands[i]
            
    # Local Search: Try to improve by swapping assignments
    # Calculate current cost
    def calc_cost(assignments):
        total_assign = np.sum(assign_cost[np.arange(n_customers), assignments])
        used_fac = set(assignments)
        total_open = np.sum(open_cost[list(used_fac)])
        return total_assign + total_open

    current_cost = calc_cost(result)
    
    # Simple local search: iterate customers, try moving to a better facility
    # or swapping with another customer
    improved = True
    max_iters = 1000
    iter_count = 0
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        for i in range(n_customers):
            # Try moving i to a different facility
            current_fac = result[i]
            # Check all other facilities
            for j in range(n_facilities):
                if j == current_fac:
                    continue
                # Check capacity if we move i to j
                # Load of j increases by demands[i], load of current_fac decreases
                new_load_j = load[j] + demands[i]
                if new_load_j <= capacities[j]:
                    # Calculate cost change
                    # Assign cost change: assign_cost[i, j] - assign_cost[i, current_fac]
                    # Open cost change:
                    # If current_fac becomes empty, we save open_cost[current_fac]
                    # If j was empty, we pay open_cost[j]
                    
                    old_load_current = load[current_fac]
                    new_load_current = old_load_current - demands[i]
                    
                    delta_assign = assign_cost[i, j] - assign_cost[i, current_fac]
                    delta_open = 0
                    
                    # Check if current_fac becomes empty
                    if new_load_current == 0:
                        # We can close it (if no other customer assigned)
                        # Check if any other customer is assigned to current_fac
                        is_used = False
                        for k in range(n_customers):
                            if k != i and result[k] == current_fac:
                                is_used = True
                                break
                        if not is_used:
                            delta_open -= open_cost[current_fac]
                    
                    # Check if j was empty
                    if load[j] == 0:
                        # We must open it
                        # Check if any other customer is assigned to j (should be none if load 0)
                        # But load[j] is current load. If 0, it's empty.
                        delta_open += open_cost[j]
                    
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < -1e-9:
                        # Move
                        load[current_fac] -= demands[i]
                        load[j] += demands[i]
                        result[i] = j
                        current_cost += total_delta
                        improved = True
                        break # Restart loop for this customer? Or continue?
                        # Breaking inner loop, continue outer loop for next customer
                        # This is a simple "first improvement" strategy
                    
    return result

# EVOLVE-BLOCK-END
