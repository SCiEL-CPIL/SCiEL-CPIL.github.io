"""Program using Multi-Start Local Search for capacitated facility location."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using Multi-Start Local Search.
    
    Args:
    demands: array-like of customer demands
    capacities: array-like of facility capacities
    open_cost: array-like of facility opening costs
    assign_cost: 2D array of assignment costs
    
    Returns:
    numpy array of facility indices for each customer
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Precompute indices for efficiency
    customer_indices = np.arange(n_cust)
    
    def calc_cost(assignment, load):
        """Calculate total cost given assignment and load."""
        # Check capacity validity
        if np.any(load > capacities + 1e-9):
            return np.inf
            
        cost = 0.0
        # Open costs: sum open_cost[f] for f where load[f] > 0
        # Using a mask is faster than a loop
        active_facs = load > 0
        cost += np.sum(open_cost[active_facs])
        
        # Assignment costs
        cost += np.sum(assign_cost[customer_indices, assignment])
        return cost

    def greedy_init(seed=None):
        """
        Generate a valid initial solution using greedy heuristic.
        If seed is provided, we can introduce slight randomization in tie-breaking
        or order to create diverse starts, but standard greedy is a strong baseline.
        """
        res = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        
        # Sort customers by demand descending
        order = np.argsort(-demands)
        
        for i in order:
            # Sort facilities by cost for this customer
            # Add small random noise to break ties and create diversity if needed?
            # For now, deterministic greedy is robust.
            fac_order = np.argsort(assign_cost[i])
            
            assigned = False
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    res[i] = f
                    assigned = True
                    break
            
            if not assigned:
                # Fallback: should not happen if instance is feasible
                # Find facility with most remaining capacity
                rem_cap = capacities - load
                best_f = np.argmax(rem_cap)
                if load[best_f] + demands[i] <= capacities[best_f]:
                    load[best_f] += demands[i]
                    res[i] = best_f
                else:
                    # Instance infeasible? Task implies feasible. 
                    # Force assign to first available or raise error.
                    # For safety, assign to first facility (invalid result, but prevents crash)
                    res[i] = 0
                    load[0] += demands[i]
                    
        return res, load

    def local_search(assignment, load, max_iter=5000):
        """
        Improve assignment using Relocate and Swap moves.
        Strictly descending hill climbing.
        """
        current_cost = calc_cost(assignment, load)
        if np.isinf(current_cost):
            return assignment, current_cost
            
        no_improve_count = 0
        max_no_improve = 100
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
                
            improved = False
            
            # 1. Relocate Move: Move one customer to another facility
            # Randomly pick a customer
            i = np.random.randint(n_cust)
            current_f = assignment[i]
            
            # Try to move to the best alternative facility
            # Calculate delta for moving i from current_f to f_new
            # Delta = (assign_cost[i, f_new] - assign_cost[i, current_f]) 
            #        + (open_cost[f_new] if load[f_new]==0 else 0) 
            #        - (open_cost[current_f] if load[current_f]==demands[i] else 0)
            
            best_delta = 0.0 # Only accept if strictly better (delta < 0)
            best_f_new = -1
            
            # We only need to check facilities that have enough capacity
            # And potentially close to current_f in cost
            # For efficiency, check all facilities but break early if possible?
            # n_fac is small (20), so checking all is fine.
            
            for f_new in range(n_fac):
                if f_new == current_f:
                    continue
                
                # Capacity check
                if load[f_new] + demands[i] > capacities[f_new] + 1e-9:
                    continue
                
                # Calculate cost change
                delta_assign = assign_cost[i, f_new] - assign_cost[i, current_f]
                
                delta_open = 0.0
                # Opening cost for f_new
                if load[f_new] == 0:
                    delta_open += open_cost[f_new]
                # Closing cost for current_f (if it becomes empty)
                if load[current_f] == demands[i]:
                    delta_open -= open_cost[current_f]
                
                total_delta = delta_assign + delta_open
                
                if total_delta < best_delta:
                    best_delta = total_delta
                    best_f_new = f_new
            
            if best_f_new != -1:
                # Apply move
                load[current_f] -= demands[i]
                load[best_f_new] += demands[i]
                assignment[i] = best_f_new
                current_cost += best_delta
                improved = True
                no_improve_count = 0
            else:
                # Try Swap Move: Swap two customers
                if np.random.random() < 0.5:
                    i, j = np.random.choice(n_cust, size=2, replace=False)
                    f_i = assignment[i]
                    f_j = assignment[j]
                    
                    if f_i != f_j:
                        # Check capacity after swap
                        # load[f_i] - d_i + d_j <= cap[f_i]
                        # load[f_j] - d_j + d_i <= cap[f_j]
                        if (load[f_i] - demands[i] + demands[j] <= capacities[f_i] + 1e-9 and
                            load[f_j] - demands[j] + demands[i] <= capacities[f_j] + 1e-9):
                            
                            # Calculate delta
                            # Assign cost change
                            delta_assign = (assign_cost[i, f_j] - assign_cost[i, f_i]) + \
                                           (assign_cost[j, f_i] - assign_cost[j, f_j])
                            
                            # Open cost change
                            # Does f_i become empty? Does f_j become empty?
                            # Does f_i start empty? No, it has i. Does f_j start empty? No, it has j.
                            # So we only check if they become empty.
                            new_load_fi = load[f_i] - demands[i] + demands[j]
                            new_load_fj = load[f_j] - demands[j] + demands[i]
                            
                            delta_open = 0.0
                            if new_load_fi == 0:
                                delta_open -= open_cost[f_i]
                            if new_load_fj == 0:
                                delta_open -= open_cost[f_j]
                            
                            total_delta = delta_assign + delta_open
                            
                            if total_delta < 0:
                                # Apply swap
                                load[f_i] = new_load_fi
                                load[f_j] = new_load_fj
                                assignment[i], assignment[j] = f_j, f_i
                                current_cost += total_delta
                                improved = True
                                no_improve_count = 0
                else:
                    no_improve_count += 1
            
            if not improved:
                no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                break
                
        return assignment, current_cost

    # Multi-Start Strategy
    best_assign = None
    best_cost = np.inf
    
    # Run multiple greedy initializations with different random seeds/shuffles
    # Since the problem is small, we can afford several restarts.
    n_starts = 10
    
    for s in range(n_starts):
        if time.time() > deadline:
            break
            
        # Create diverse starts by shuffling the order of customers with equal demand?
        # Or just run standard greedy. Standard greedy is deterministic.
        # To get diversity, we can perturb the assignment costs slightly for the init phase only?
        # Or simply run the same greedy, but local search will move it.
        # Better: Randomized Greedy.
        
        res, load = greedy_init()
        
        # Run local search
        res, cost = local_search(res, load)
        
        if cost < best_cost:
            best_cost = cost
            best_assign = res.copy()
            
        # If we find a very good solution, we might still want to keep going
        # to ensure we haven't missed a better basin.
        
    # Final check: If best_assign is None (shouldn't happen), fallback to simple greedy
    if best_assign is None:
        best_assign, _ = greedy_init()
        
    return best_assign

# EVOLVE-BLOCK-END
