"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_cost(result, demands, capacities, open_cost, assign_cost):
    """Compute the total cost of a given assignment."""
    if result is None:
        return np.inf
    
    n_customers = len(result)
    n_facilities = len(capacities)
    
    # Check validity
    load = np.zeros(n_facilities, dtype=float)
    for i in range(n_customers):
        f = result[i]
        if f < 0 or f >= n_facilities:
            return np.inf
        load[f] += demands[i]
        
    for f in range(n_facilities):
        if load[f] > capacities[f] + 1e-9:
            return np.inf
            
    total_cost = 0.0
    for f in range(n_facilities):
        if load[f] > 0:
            total_cost += open_cost[f]
            
    total_cost += np.sum(assign_cost[np.arange(n_customers), result])
    return total_cost

def _greedy_init(demands, capacities, open_cost, assign_cost):
    """
    Greedy initialization: 
    Sort customers by demand descending.
    Assign to facility with min (assign_cost + marginal_open_cost).
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities, dtype=float)
    open_status = np.zeros(n_facilities, dtype=bool)
    
    order = np.argsort(-demands)
    
    for i in order:
        best_f = -1
        best_cost = np.inf
        for f in range(n_facilities):
            if load[f] + demands[i] <= capacities[f]:
                inc_open = open_cost[f] if not open_status[f] else 0.0
                total_inc = assign_cost[i, f] + inc_open
                if total_inc < best_cost:
                    best_cost = total_inc
                    best_f = f
        
        if best_f != -1:
            result[i] = best_f
            load[best_f] += demands[i]
            open_status[best_f] = True
        else:
            # Fallback
            rem_cap = capacities - load
            best_f = np.argmax(rem_cap)
            result[i] = best_f
            load[best_f] += demands[i]
            open_status[best_f] = True
            
    return result

def _local_search(result, demands, capacities, open_cost, assign_cost, time_limit=5):
    """
    Local search with single moves and swaps.
    """
    n_customers = len(result)
    n_facilities = len(capacities)
    
    result = result.copy()
    load = np.zeros(n_facilities, dtype=float)
    for i in range(n_customers):
        load[result[i]] += demands[i]
        
    current_cost = _compute_cost(result, demands, capacities, open_cost, assign_cost)
    if np.isinf(current_cost):
        return result
        
    start_time = time.time()
    improved = True
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # 1. Single Moves
        for i in range(n_customers):
            f_old = result[i]
            d_i = demands[i]
            best_f = -1
            best_delta = 0.0
            
            for f_new in range(n_facilities):
                if f_new == f_old:
                    continue
                if load[f_new] + d_i > capacities[f_new] + 1e-9:
                    continue
                
                delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
                delta_open = 0.0
                if load[f_new] == 0:
                    delta_open += open_cost[f_new]
                if load[f_old] == d_i:
                    delta_open -= open_cost[f_old]
                    
                total_delta = delta_assign + delta_open
                
                if total_delta < best_delta:
                    best_delta = total_delta
                    best_f = f_new
                    
            if best_f != -1 and best_delta < -1e-9:
                load[f_old] -= d_i
                load[best_f] += d_i
                result[i] = best_f
                current_cost += best_delta
                improved = True
        
        # 2. Swaps
        for i in range(n_customers):
            f_i = result[i]
            d_i = demands[i]
            for k in range(i + 1, n_customers):
                f_k = result[k]
                d_k = demands[k]
                
                if f_i == f_k:
                    continue
                
                new_load_i = load[f_i] - d_i + d_k
                new_load_k = load[f_k] - d_k + d_i
                
                if new_load_i > capacities[f_i] + 1e-9 or new_load_k > capacities[f_k] + 1e-9:
                    continue
                
                delta_assign = (assign_cost[i, f_k] + assign_cost[k, f_i]) - (assign_cost[i, f_i] + assign_cost[k, f_k])
                delta_open = 0.0
                
                if new_load_i == 0:
                    delta_open -= open_cost[f_i]
                if new_load_k == 0:
                    delta_open -= open_cost[f_k]
                
                # Check if any facility opens? 
                # In a swap, no facility opens if it was already open (which it is, since i and k are there).
                # If f_i was only i, and k moves in, it stays open.
                # If f_i was only i, and k does NOT move in (impossible in swap), it closes.
                # Wait, if f_i == f_k, we skip.
                # So f_i has i, f_k has k.
                # After swap: f_i has k, f_k has i.
                # Neither facility becomes empty unless d_i=0 or d_k=0, which is not the case (min demand 5).
                # So delta_open is 0 unless one of the facilities had ONLY that customer and the other customer moves there?
                # No, if f_i had only i, load[f_i]=d_i. New load = d_k. If d_k>0, it stays open.
                # So delta_open is always 0 for swaps between two non-empty facilities?
                # Yes, because both facilities remain non-empty.
                
                total_delta = delta_assign + delta_open
                
                if total_delta < -1e-9:
                    load[f_i] = new_load_i
                    load[f_k] = new_load_k
                    result[i] = f_k
                    result[k] = f_i
                    current_cost += total_delta
                    improved = True
                    break # Restart inner loop
                    
    return result

def _simulated_annealing(result, demands, capacities, open_cost, assign_cost, time_limit=20):
    """
    Simulated Annealing to escape local optima.
    """
    n_customers = len(result)
    n_facilities = len(capacities)
    
    current_sol = result.copy()
    load = np.zeros(n_facilities, dtype=float)
    for i in range(n_customers):
        load[current_sol[i]] += demands[i]
        
    current_cost = _compute_cost(current_sol, demands, capacities, open_cost, assign_cost)
    if np.isinf(current_cost):
        return result

    best_sol = current_sol.copy()
    best_cost = current_cost
    
    # Parameters
    T = 100.0 # Initial temperature (cost scale is ~1000-10000)
    alpha = 0.9999 # Cooling rate
    T_min = 0.1
    
    start_time = time.time()
    
    while T > T_min and (time.time() - start_time) < time_limit:
        # Generate a random move
        move_type = np.random.rand()
        
        if move_type < 0.7:
            # Single move
            i = np.random.randint(n_customers)
            f_old = current_sol[i]
            d_i = demands[i]
            
            # Find feasible facilities
            feasible = np.where(load + d_i <= capacities)[0]
            feasible = feasible[feasible != f_old]
            
            if len(feasible) == 0:
                continue
                
            f_new = feasible[np.random.randint(len(feasible))]
            
            # Calculate delta
            delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
            delta_open = 0.0
            if load[f_new] == 0:
                delta_open += open_cost[f_new]
            if load[f_old] == d_i:
                delta_open -= open_cost[f_old]
                
            delta = delta_assign + delta_open
            
            # Acceptance
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                # Accept
                load[f_old] -= d_i
                load[f_new] += d_i
                current_sol[i] = f_new
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_sol = current_sol.copy()
                    
        else:
            # Swap
            i = np.random.randint(n_customers)
            k = np.random.randint(n_customers)
            if i == k:
                continue
                
            f_i = current_sol[i]
            f_k = current_sol[k]
            if f_i == f_k:
                continue
                
            d_i = demands[i]
            d_k = demands[k]
            
            new_load_i = load[f_i] - d_i + d_k
            new_load_k = load[f_k] - d_k + d_i
            
            if new_load_i > capacities[f_i] + 1e-9 or new_load_k > capacities[f_k] + 1e-9:
                continue
                
            delta_assign = (assign_cost[i, f_k] + assign_cost[k, f_i]) - (assign_cost[i, f_i] + assign_cost[k, f_k])
            # Delta open is 0 as reasoned before
            
            delta = delta_assign
            
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                load[f_i] = new_load_i
                load[f_k] = new_load_k
                current_sol[i] = f_k
                current_sol[k] = f_i
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_sol = current_sol.copy()
                    
        T *= alpha
        
    return best_sol

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 145 # 5s margin
    
    # 1. Greedy + Local Search
    sol1 = _greedy_init(demands, capacities, open_cost, assign_cost)
    sol1 = _local_search(sol1, demands, capacities, open_cost, assign_cost, time_limit=2)
    cost1 = _compute_cost(sol1, demands, capacities, open_cost, assign_cost)
    
    best_sol = sol1
    best_cost = cost1
    
    # 2. Simulated Annealing
    # Use remaining time for SA
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    
    if remaining > 2:
        sa_sol = _simulated_annealing(best_sol, demands, capacities, open_cost, assign_cost, time_limit=remaining - 1)
        sa_cost = _compute_cost(sa_sol, demands, capacities, open_cost, assign_cost)
        
        if sa_cost < best_cost:
            best_sol = sa_sol
            best_cost = sa_cost
            
    return best_sol
# EVOLVE-BLOCK-END
