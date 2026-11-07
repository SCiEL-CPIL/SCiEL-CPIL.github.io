"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using Lagrangian relaxation with
    efficient greedy construction and local search refinement.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Try multiple random restarts with greedy + local search
    best_result = None
    best_cost = np.inf
    
    # Number of restarts based on time budget
    n_restarts = 50
    
    for restart in range(n_restarts):
        result = _greedy_with_randomization(demands, capacities, open_cost, assign_cost, restart)
        # Refine with local search
        result = _local_search(result, demands, capacities, open_cost, assign_cost, max_iters=200)
        
        # Calculate cost
        cost = _calc_cost(result, demands, capacities, open_cost, assign_cost)
        
        if cost < best_cost:
            best_cost = cost
            best_result = result.copy()
    
    return best_result

def _greedy_with_randomization(demands, capacities, open_cost, assign_cost, seed):
    """
    Greedy assignment with randomization to explore different solutions.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    rng = np.random.default_rng(seed)
    
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    
    # Sort customers by demand descending, with some randomization
    order = np.argsort(-demands)
    
    # Add random tie-breaking for customers with similar demands
    for i in range(len(order) - 1):
        if demands[order[i]] - demands[order[i+1]] < 3:
            if rng.random() < 0.3:
                order[i], order[i+1] = order[i+1], order[i]
    
    for i in order:
        # Find feasible facilities
        feasible = []
        for j in range(n_facilities):
            if load[j] + demands[i] <= capacities[j]:
                # Add random perturbation to cost for diversity
                perturbed_cost = assign_cost[i, j] + rng.random() * 10
                feasible.append((perturbed_cost, j))
        
        if not feasible:
            # Emergency: pick facility with max remaining capacity
            rem_cap = capacities - load
            best_fac = np.argmax(rem_cap)
            result[i] = best_fac
            load[best_fac] += demands[i]
        else:
            # Sort by perturbed cost
            feasible.sort()
            best_fac = feasible[0][1]
            result[i] = best_fac
            load[best_fac] += demands[i]
    
    return result

def _local_search(result, demands, capacities, open_cost, assign_cost, max_iters=100):
    """
    Local search to improve assignment by moving customers to better facilities.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        load[result[i]] += demands[i]
    
    # Track which facilities are open
    open_facilities = np.zeros(n_facilities, dtype=bool)
    for i in range(n_customers):
        open_facilities[result[i]] = True
    
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        
        for i in range(n_customers):
            current_fac = result[i]
            best_delta = 0
            best_fac = current_fac
            
            for j in range(n_facilities):
                if j == current_fac:
                    continue
                
                # Check if moving i to j is feasible
                new_load_j = load[j] + demands[i]
                if new_load_j > capacities[j]:
                    continue
                
                # Calculate cost delta
                delta_assign = assign_cost[i, j] - assign_cost[i, current_fac]
                delta_open = 0
                
                # Check if current_fac becomes empty
                new_load_current = load[current_fac] - demands[i]
                if new_load_current == 0:
                    # Facility becomes empty, save open cost
                    delta_open -= open_cost[current_fac]
                
                # Check if j was empty
                if load[j] == 0:
                    # Facility needs to be opened, pay open cost
                    delta_open += open_cost[j]
                
                total_delta = delta_assign + delta_open
                
                if total_delta < best_delta - 1e-9:
                    best_delta = total_delta
                    best_fac = j
            
            # Apply best move if it improves
            if best_delta < -1e-9 and best_fac != current_fac:
                # Update loads
                load[current_fac] -= demands[i]
                load[best_fac] += demands[i]
                
                # Update open facilities
                if load[current_fac] == 0:
                    open_facilities[current_fac] = False
                if load[best_fac] == 0:
                    open_facilities[best_fac] = True
                
                result[i] = best_fac
                improved = True
    
    return result

def _calc_cost(result, demands, capacities, open_cost, assign_cost):
    """
    Calculate total cost of an assignment.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Assignment cost
    assign_total = np.sum(assign_cost[np.arange(n_customers), result])
    
    # Opening cost
    open_total = 0
    for j in range(n_facilities):
        if np.any(result == j):
            open_total += open_cost[j]
    
    return assign_total + open_total

# EVOLVE-BLOCK-END
