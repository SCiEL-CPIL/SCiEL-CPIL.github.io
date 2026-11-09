import numpy as np
import time
import random

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using a heuristic with local search.
    
    Args:
    demands: array-like of customer demands
    capacities: array-like of facility capacities
    open_cost: array-like of facility opening costs
    assign_cost: 2D array of assignment costs
    
    Returns:
    numpy array of facility indices for each customer
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
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
        """Greedy initialization with randomization."""
        random.seed(seed)
        res = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        
        # Sort customers by demand descending to place large items first
        order = np.argsort(-demands)
        
        for i in order:
            # Get facilities sorted by assignment cost
            fac_costs = assign_cost[i]
            
            # Add noise to break ties and explore
            noise = np.random.uniform(0, 1, n_fac) * np.std(fac_costs)
            noisy_costs = fac_costs + noise
            fac_order_noisy = np.argsort(noisy_costs)
            
            assigned = False
            for f in fac_order_noisy:
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    res[i] = f
                    load[f] += demands[i]
                    assigned = True
                    break
            
            if not assigned:
                # Fallback: facility with most remaining space
                space = capacities - load
                max_space = np.max(space)
                candidates = np.where(space == max_space)[0]
                f = random.choice(candidates.tolist())
                res[i] = f
                load[f] += demands[i]
                
        return res

    def local_search(res, load, max_iter=5000):
        """Local search with relocate and swap moves."""
        improved = True
        it = 0
        while improved and it < max_iter:
            if time.time() > deadline:
                break
            improved = False
            it += 1
            
            # --- Phase 1: Relocate Moves ---
            for i in np.random.permutation(n_cust):
                current_f = res[i]
                best_delta = 1e-9
                best_f = -1
                
                curr_cost_i = assign_cost[i, current_f]
                
                for f_new in range(n_fac):
                    if f_new == current_f:
                        continue
                    
                    if load[f_new] + demands[i] > capacities[f_new] + 1e-9:
                        continue
                    
                    delta = assign_cost[i, f_new] - curr_cost_i
                    
                    if load[f_new] == 0:
                        delta += open_cost[f_new]
                    
                    if load[current_f] == demands[i]:
                        delta -= open_cost[current_f]
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_f = f_new
                
                if best_f != -1:
                    load[current_f] -= demands[i]
                    load[best_f] += demands[i]
                    res[i] = best_f
                    improved = True
            
            # --- Phase 2: Swap Moves ---
            if it % 5 == 0:
                swapped = False
                pairs = []
                for i in range(n_cust):
                    for j in range(i+1, n_cust):
                        pairs.append((i, j))
                random.shuffle(pairs)
                
                for i, j in pairs[:200]:
                    if time.time() > deadline:
                        break
                        
                    f_i = res[i]
                    f_j = res[j]
                    
                    if f_i == f_j:
                        continue
                    
                    if load[f_i] - demands[i] + demands[j] > capacities[f_i] + 1e-9:
                        continue
                    if load[f_j] - demands[j] + demands[i] > capacities[f_j] + 1e-9:
                        continue
                    
                    delta = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
                    
                    if delta < -1e-9:
                        load[f_i] += demands[j] - demands[i]
                        load[f_j] += demands[i] - demands[j]
                        res[i], res[j] = f_j, f_i
                        improved = True
                        swapped = True
                        break
        
        return res

    def lagrangian_init(max_iter=100):
        """
        Use Lagrangian relaxation to generate diverse, high-quality initial solutions.
        This acts as a structural change to the initialization phase, leveraging
        dual information to guide the assignment.
        """
        multipliers = np.zeros(n_fac)
        best_res = None
        best_load = None
        best_score = np.inf
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Compute effective costs
            eff_costs = assign_cost + np.outer(demands, multipliers)
            
            # Greedy assignment based on effective costs
            res = np.zeros(n_cust, dtype=int)
            load = np.zeros(n_fac)
            order = np.argsort(-demands)
            
            for i in order:
                min_cost = np.inf
                best_f = -1
                for f in range(n_fac):
                    if load[f] + demands[i] <= capacities[f] + 1e-9:
                        if eff_costs[i, f] < min_cost:
                            min_cost = eff_costs[i, f]
                            best_f = f
                
                if best_f >= 0:
                    res[i] = best_f
                    load[best_f] += demands[i]
                else:
                    space = capacities - load
                    best_f = np.argmax(space)
                    res[i] = best_f
                    load[best_f] += demands[i]
            
            # Check feasibility
            if np.any(load > capacities + 1e-9):
                # Infeasible, update multipliers
                subgrad = capacities - load
                step = 0.5 / (it + 1)
                multipliers = np.maximum(0, multipliers + step * subgrad)
            else:
                # Feasible
                score = compute_score(res)
                if score < best_score:
                    best_score = score
                    best_res = res.copy()
                    best_load = load.copy()
                
                # Update multipliers to encourage feasibility
                subgrad = capacities - load
                step = 0.5 / (it + 1)
                multipliers = np.maximum(0, multipliers + step * subgrad)
        
        if best_res is not None:
            return best_res, best_load
        else:
            # Fallback to greedy
            res = greedy_init(0)
            load = np.zeros(n_fac)
            np.add.at(load, res, demands)
            return res, load

    best_assign = None
    best_score = np.inf
    
    # Run multiple iterations with different seeds
    # Use Lagrangian for the first few restarts to get diverse, high-quality starts
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        if r < 5:
            # Use Lagrangian relaxation for the first 5 restarts
            res, load = lagrangian_init(max_iter=50)
        else:
            # Use standard greedy for the rest
            res = greedy_init(r)
            load = np.zeros(n_fac)
            np.add.at(load, res, demands)
            
            # Ensure feasibility
            if np.any(load > capacities + 1e-9):
                # Repair
                for i in range(n_cust):
                    if load[res[i]] > capacities[res[i]] + 1e-9:
                        # Find a facility with space
                        for f in range(n_fac):
                            if load[f] + demands[i] <= capacities[f] + 1e-9:
                                load[res[i]] -= demands[i]
                                load[f] += demands[i]
                                res[i] = f
                                break
        
        local_search(res, load, max_iter=200)
        
        score = compute_score(res)
        if score < best_score:
            best_score = score
            best_assign = res.copy()
            
    return best_assign.astype(int)
