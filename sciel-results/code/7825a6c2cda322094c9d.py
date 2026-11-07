"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _calculate_objective(assignment, demands, capacities, open_cost, assign_cost):
    """Calculate the total cost of an assignment."""
    n_cust = len(assignment)
    n_fac = len(capacities)
    
    # Check validity
    load = np.zeros(n_fac)
    for i in range(n_cust):
        f = assignment[i]
        if f < 0 or f >= n_fac:
            return float('inf')
        load[f] += demands[i]
        
    if np.any(load > capacities + 1e-9):
        return float('inf')
        
    open_sum = 0.0
    used = np.zeros(n_fac, dtype=bool)
    for f in assignment:
        used[f] = True
    open_sum = np.sum(open_cost[used])
    
    assign_sum = np.sum(assign_cost[np.arange(n_cust), assignment])
    
    return open_sum + assign_sum

def _greedy_init(demands, capacities, open_cost, assign_cost):
    """
    Greedy initialization: Sort customers by demand descending.
    Assign to the facility that minimizes marginal cost.
    Marginal cost = assign_cost[i, f] + (open_cost[f] if load[f]==0 else 0)
    This encourages opening few facilities and filling them.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    assignment = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    
    # Sort indices by demand descending
    order = np.argsort(-demands)
    
    for i in order:
        best_f = -1
        best_cost = float('inf')
        for f in range(n_fac):
            if load[f] + demands[i] > capacities[f]:
                continue
            cost = assign_cost[i, f]
            if load[f] == 0:
                cost += open_cost[f]
            if cost < best_cost:
                best_cost = cost
                best_f = f
        
        if best_f == -1:
            # Should not happen if capacities are sufficient, but fallback
            # Find any feasible facility
            for f in range(n_fac):
                if load[f] + demands[i] <= capacities[f]:
                    best_f = f
                    break
            if best_f == -1:
                return None # Infeasible instance?
        
        assignment[i] = best_f
        load[best_f] += demands[i]
        
    return assignment

def _local_search(assignment, demands, capacities, open_cost, assign_cost, time_limit=None, start_time=None):
    """
    Local search with moves and swaps.
    If time_limit is provided, it will run until time runs out.
    Otherwise, it runs until no improvement is found (greedy LS).
    """
    n_cust = len(assignment)
    n_fac = len(capacities)
    
    # Convert to list for easier modification if needed, but numpy is fine
    assignment = assignment.copy()
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[assignment[i]] += demands[i]
    
    current_obj = _calculate_objective(assignment, demands, capacities, open_cost, assign_cost)
    if np.isinf(current_obj):
        return assignment
    
    best_obj = current_obj
    best_assignment = assignment.copy()
    
    if time_limit is None:
        # Greedy Local Search: Keep improving until stuck
        improved = True
        while improved:
            improved = False
            # 1. Try Moves
            for i in range(n_cust):
                old_f = assignment[i]
                for new_f in range(n_fac):
                    if new_f == old_f:
                        continue
                    if load[new_f] + demands[i] > capacities[new_f]:
                        continue
                    
                    # Calculate delta
                    delta_assign = assign_cost[i, new_f] - assign_cost[i, old_f]
                    delta_open = 0.0
                    if load[old_f] - demands[i] == 0:
                        delta_open -= open_cost[old_f]
                    if load[new_f] == 0:
                        delta_open += open_cost[new_f]
                    
                    delta = delta_assign + delta_open
                    if delta < -1e-9:
                        # Apply move
                        load[old_f] -= demands[i]
                        load[new_f] += demands[i]
                        assignment[i] = new_f
                        current_obj += delta
                        if current_obj < best_obj:
                            best_obj = current_obj
                            best_assignment = assignment.copy()
                        improved = True
                        break # Restart loop for this customer? Or just break inner? 
                        # Breaking inner is standard steepest descent. 
                        # Let's break inner and continue to next customer to be fast.
            
            # 2. Try Swaps (only if we want stronger LS, but for greedy init polish, moves are often enough)
            # We'll stick to moves for the fast greedy phase to save time for SA later.
            
    else:
        # Simulated Annealing / Time-bounded Search
        # We will mix random moves and random swaps to escape local optima
        T = 100.0 # Initial temperature
        cooling_rate = 0.999
        min_T = 0.1
        
        # Precompute some structures? Not really needed for N=100.
        
        while time.time() - start_time < time_limit:
            # Pick a random operation: Move (70%) or Swap (30%)
            if random.random() < 0.7:
                # Random Move
                i = random.randint(0, n_cust - 1)
                old_f = assignment[i]
                new_f = random.randint(0, n_fac - 1)
                if new_f == old_f:
                    continue
                if load[new_f] + demands[i] > capacities[new_f]:
                    continue
                
                delta_assign = assign_cost[i, new_f] - assign_cost[i, old_f]
                delta_open = 0.0
                if load[old_f] - demands[i] == 0:
                    delta_open -= open_cost[old_f]
                if load[new_f] == 0:
                    delta_open += open_cost[new_f]
                
                delta = delta_assign + delta_open
            else:
                # Random Swap
                i = random.randint(0, n_cust - 1)
                k = random.randint(0, n_cust - 1)
                if i == k:
                    continue
                f_i = assignment[i]
                f_k = assignment[k]
                if f_i == f_k:
                    continue
                
                # Check capacity for swap
                # i moves to f_k, k moves to f_i
                # load[f_k] - d_k + d_i <= cap[f_k]
                # load[f_i] - d_i + d_k <= cap[f_i]
                if load[f_k] - demands[k] + demands[i] > capacities[f_k]:
                    continue
                if load[f_i] - demands[i] + demands[k] > capacities[f_i]:
                    continue
                
                delta_assign = (assign_cost[i, f_k] + assign_cost[k, f_i]) - (assign_cost[i, f_i] + assign_cost[k, f_k])
                
                delta_open = 0.0
                # Check if f_i becomes empty
                new_load_fi = load[f_i] - demands[i] + demands[k]
                if new_load_fi == 0:
                    delta_open -= open_cost[f_i]
                # Check if f_k becomes empty
                new_load_fk = load[f_k] - demands[k] + demands[i]
                if new_load_fk == 0:
                    delta_open -= open_cost[f_k]
                
                # Note: We don't add open cost if a facility was empty and becomes non-empty?
                # In a swap, f_i and f_k are both currently non-empty (they hold i and k).
                # So they cannot "become" used, only "become" empty.
                
                delta = delta_assign + delta_open
                
                # Apply swap logic for acceptance
                if delta < 0 or random.random() < np.exp(-delta / T):
                    # Apply swap
                    load[f_i] = new_load_fi
                    load[f_k] = new_load_fk
                    assignment[i] = f_k
                    assignment[k] = f_i
                    current_obj += delta
                    if current_obj < best_obj:
                        best_obj = current_obj
                        best_assignment = assignment.copy()
                    T = max(min_T, T * cooling_rate)
                    continue # Skip the move logic below
                
                # If swap rejected, do nothing to loads/assignment
                T = max(min_T, T * cooling_rate)
                continue

            # Move Logic (if not swapped)
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Apply move
                load[old_f] -= demands[i]
                load[new_f] += demands[i]
                assignment[i] = new_f
                current_obj += delta
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_assignment = assignment.copy()
            
            T = max(min_T, T * cooling_rate)
            
    return best_assignment

def assign(demands, capacities, open_cost, assign_cost):
    demands = np.array(demands)
    capacities = np.array(capacities)
    open_cost = np.array(open_cost)
    assign_cost = np.array(assign_cost)
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    start_time = time.time()
    time_limit = 110.0 # 10s buffer
    
    # 1. Generate initial solution
    # Try a few greedy inits to pick the best start
    best_init = None
    best_init_obj = float('inf')
    
    for _ in range(10):
        init = _greedy_init(demands, capacities, open_cost, assign_cost)
        if init is not None:
            obj = _calculate_objective(init, demands, capacities, open_cost, assign_cost)
            if obj < best_init_obj:
                best_init_obj = obj
                best_init = init
    
    if best_init is None:
        # Fallback: Assign all to facility 0? Might be infeasible.
        # Try to find any feasible assignment.
        # Given the problem constraints, this shouldn't happen.
        return np.zeros(n_cust, dtype=int)
    
    # 2. Polish with fast local search (moves only)
    polished = _local_search(best_init, demands, capacities, open_cost, assign_cost, time_limit=None)
    
    # 3. Simulated Annealing / Bounded Search
    # Check time remaining
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    
    if remaining > 1.0:
        # Run SA for the remaining time
        # We pass the polished solution as start
        final_result = _local_search(polished, demands, capacities, open_cost, assign_cost, time_limit=remaining, start_time=start_time)
    else:
        final_result = polished
        
    return final_result
# EVOLVE-BLOCK-END
