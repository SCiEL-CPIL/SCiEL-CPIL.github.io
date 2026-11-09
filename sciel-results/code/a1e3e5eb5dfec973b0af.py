import numpy as np
import time
import random

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using tabu search with shake diversification.
    
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
    
    def compute_score(res):
        """Compute total cost of an assignment."""
        load = np.zeros(n_fac)
        np.add.at(load, res, demands)
        if np.any(load > capacities + 1e-9):
            return np.inf
        used = load > 0
        return np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(n_cust), res])
    
    def greedy_init(seed):
        """Greedy initialization: assign largest demands first to feasible facilities."""
        rng = np.random.RandomState(seed)
        res = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        
        # Sort by demand descending
        order = np.argsort(-demands)
        
        for i in order:
            # Try facilities in order of assignment cost, with small random perturbation
            costs = assign_cost[i] + rng.uniform(0, 0.5 * np.std(assign_cost[i]), n_fac)
            fac_order = np.argsort(costs)
            
            assigned = False
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    res[i] = f
                    load[f] += demands[i]
                    assigned = True
                    break
            
            if not assigned:
                # Fallback: facility with most remaining space
                space = capacities - load
                f = np.argmax(space)
                res[i] = f
                load[f] += demands[i]
                
        return res, load

    def tabu_search(res, load, tabu_len=10, max_iter=500):
        """Tabu search with relocate and swap moves."""
        rng = np.random.RandomState()
        
        # Tabu list: (customer, facility) pairs that are forbidden for tabu_len iterations
        tabu_list = {}  # (cust, fac) -> iteration when tabu expires
        current_iter = 0
        
        # Track best solution
        best_res = res.copy()
        best_load = load.copy()
        best_score = compute_score(res)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            current_iter += 1
            
            # Clean up expired tabu entries
            expired_keys = [k for k, exp in tabu_list.items() if exp <= current_iter]
            for k in expired_keys:
                del tabu_list[k]
            
            best_move = None
            best_delta = 1e-9  # Threshold for improvement
            
            # --- Relocate moves ---
            for i in rng.permutation(n_cust):
                current_f = res[i]
                curr_cost_i = assign_cost[i, current_f]
                
                for f_new in range(n_fac):
                    if f_new == current_f:
                        continue
                    
                    # Check tabu
                    if (i, f_new) in tabu_list and tabu_list[(i, f_new)] > current_iter:
                        continue
                    
                    # Check capacity
                    if load[f_new] + demands[i] > capacities[f_new] + 1e-9:
                        continue
                    
                    # Calculate delta
                    delta = assign_cost[i, f_new] - curr_cost_i
                    
                    # Opening cost changes
                    if load[f_new] == 0:
                        delta += open_cost[f_new]
                    if load[current_f] == demands[i]:
                        delta -= open_cost[current_f]
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_move = ('relocate', i, f_new)
            
            # --- Swap moves (sampled) ---
            if it % 3 == 0:
                # Sample 100 random pairs
                for _ in range(100):
                    i = rng.randint(n_cust)
                    j = rng.randint(n_cust)
                    if i == j:
                        continue
                    
                    f_i = res[i]
                    f_j = res[j]
                    
                    if f_i == f_j:
                        continue
                    
                    # Check tabu
                    if (i, f_j) in tabu_list and tabu_list[(i, f_j)] > current_iter:
                        continue
                    if (j, f_i) in tabu_list and tabu_list[(j, f_i)] > current_iter:
                        continue
                    
                    # Check capacity
                    if load[f_i] - demands[i] + demands[j] > capacities[f_i] + 1e-9:
                        continue
                    if load[f_j] - demands[j] + demands[i] > capacities[f_j] + 1e-9:
                        continue
                    
                    # Cost change (no opening cost change for swaps between non-empty facilities)
                    delta = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_move = ('swap', i, j)
            
            if best_move is not None:
                move_type = best_move[0]
                
                if move_type == 'relocate':
                    _, i, f_new = best_move
                    current_f = res[i]
                    
                    # Add to tabu
                    tabu_list[(i, current_f)] = current_iter + tabu_len
                    tabu_list[(i, f_new)] = current_iter + tabu_len
                    
                    # Perform move
                    load[current_f] -= demands[i]
                    load[f_new] += demands[i]
                    res[i] = f_new
                    
                else:  # swap
                    _, i, j = best_move
                    f_i = res[i]
                    f_j = res[j]
                    
                    # Add to tabu
                    tabu_list[(i, f_i)] = current_iter + tabu_len
                    tabu_list[(i, f_j)] = current_iter + tabu_len
                    tabu_list[(j, f_j)] = current_iter + tabu_len
                    tabu_list[(j, f_i)] = current_iter + tabu_len
                    
                    # Perform swap
                    load[f_i] += demands[j] - demands[i]
                    load[f_j] += demands[i] - demands[j]
                    res[i], res[j] = f_j, f_i
                
                # Update best
                score = compute_score(res)
                if score < best_score:
                    best_score = score
                    best_res = res.copy()
                    best_load = load.copy()
            else:
                # No improving move found - shake to escape local optimum
                # Randomly move 5-10 customers to random feasible facilities
                num_shake = rng.randint(5, 11)
                for _ in range(num_shake):
                    i = rng.randint(n_cust)
                    current_f = res[i]
                    
                    # Find feasible facilities
                    feasible = [f for f in range(n_fac) if load[f] + demands[i] <= capacities[f] + 1e-9]
                    if not feasible:
                        continue
                    
                    f_new = rng.choice(feasible)
                    if f_new == current_f:
                        continue
                    
                    load[current_f] -= demands[i]
                    load[f_new] += demands[i]
                    res[i] = f_new
        
        return best_res, best_load, best_score

    best_assign = None
    best_score = np.inf
    
    # Run multiple restarts with different seeds
    num_restarts = 30
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        res, load = greedy_init(r)
        
        # Ensure feasibility
        if np.any(load > capacities + 1e-9):
            # Repair: reassign overflowing customers
            for f in range(n_fac):
                while load[f] > capacities[f] + 1e-9:
                    # Find customer at f to move
                    customers_at_f = np.where(res == f)[0]
                    if len(customers_at_f) == 0:
                        break
                    i = customers_at_f[rng.randint(len(customers_at_f))]
                    
                    # Move to facility with most space
                    space = capacities - load
                    f_new = np.argmax(space)
                    if f_new == f:
                        break
                    
                    load[f] -= demands[i]
                    load[f_new] += demands[i]
                    res[i] = f_new
        
        # Run tabu search
        res, load, score = tabu_search(res, load)
        
        if score < best_score:
            best_score = score
            best_assign = res.copy()
    
    return best_assign.astype(int)
