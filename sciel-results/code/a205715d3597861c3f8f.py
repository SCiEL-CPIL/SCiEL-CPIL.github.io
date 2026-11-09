import numpy as np
import time
import random
import scipy.optimize as so

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using Lagrangian relaxation with subgradient optimization.
    
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
    
    def lagrangian_relaxation(max_iter=200):
        """
        Solve the capacitated facility location problem using Lagrangian relaxation.
        Dualize capacity constraints and use subgradient optimization.
        """
        # Multipliers for capacity constraints
        multipliers = np.zeros(n_fac)
        
        best_assign = None
        best_score = np.inf
        best_dual_bound = -np.inf
        
        # Polyhedral step size parameters
        step_mult = 2.0
        step_decay = 0.995
        step_min = 0.01
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Solve the relaxed problem for current multipliers
            # For each customer, assign to facility minimizing:
            # assign_cost[i,j] + multiplier[j] * demands[i]
            # But we also need to consider opening costs.
            
            # Approach: For each customer i, try each facility j.
            # The "effective cost" of assigning i to j is:
            # assign_cost[i,j] + multiplier[j] * demands[i]
            # Plus we need to account for opening costs.
            
            # A simpler approach: Use a greedy assignment based on effective costs,
            # then adjust multipliers based on capacity violations.
            
            # Compute effective costs
            eff_costs = assign_cost + np.outer(demands, multipliers)
            
            # Greedy assignment: sort customers by demand (descending), assign each
            # to the facility with lowest effective cost that has capacity
            res = np.zeros(n_cust, dtype=int)
            load = np.zeros(n_fac)
            
            # Sort by demand descending
            order = np.argsort(-demands)
            
            for i in order:
                # Find facility with min effective cost that has capacity
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
                    # No capacity available, assign to facility with most remaining space
                    space = capacities - load
                    best_f = np.argmax(space)
                    res[i] = best_f
                    load[best_f] += demands[i]
            
            # Check feasibility
            if np.any(load > capacities + 1e-9):
                # Infeasible, compute dual bound and update multipliers
                # Dual bound = sum of min effective costs + sum of multipliers * capacities
                dual_obj = np.sum(np.min(eff_costs, axis=1)) + np.sum(multipliers * capacities)
                best_dual_bound = max(best_dual_bound, dual_obj)
            else:
                # Feasible solution found
                score = compute_score(res)
                if score < best_score:
                    best_score = score
                    best_assign = res.copy()
                
                # Update multipliers to push towards feasibility if close to capacity
                # Use subgradient method
                for f in range(n_fac):
                    if load[f] > capacities[f] * 0.9:
                        multipliers[f] = max(0, multipliers[f] + 0.1)
                    elif load[f] < capacities[f] * 0.5:
                        multipliers[f] = max(0, multipliers[f] - 0.05)
            
            # Subgradient update
            # Subgradient of dual function with respect to multiplier lambda_j:
            # g_j = capacity_j - sum_i d_i * y_ij (for current assignment)
            # If assignment is infeasible, g_j = capacity_j - load_j < 0
            # We want to increase lambda_j when load_j > capacity_j
            
            # Compute subgradient
            subgrad = capacities - load
            
            # Step size using Polyak's rule if we have a good upper bound
            if best_score < np.inf:
                gap = best_score - best_dual_bound
                if gap > 1e-6:
                    step = step_mult * gap / (np.sum(subgrad**2) + 1e-10)
                    step = max(step_min, min(step, 1.0))
                else:
                    step = 0.0
            else:
                step = 0.5 / (it + 1)
            
            # Update multipliers (project to non-negative)
            multipliers = np.maximum(0, multipliers + step * subgrad)
            
            # Decay step
            step_mult *= step_decay
        
        return best_assign, best_score
    
    def local_search(res, load, max_iter=3000):
        """
        Local search with relocate and swap moves.
        """
        current_res = res.copy()
        current_load = load.copy()
        current_score = compute_score(current_res)
        best_res = current_res.copy()
        best_load = current_load.copy()
        best_score = current_score
        
        improved = True
        while improved and max_iter > 0:
            if time.time() > deadline:
                break
            improved = False
            max_iter -= 1
            
            # Try relocate moves
            for i in range(n_cust):
                f_old = current_res[i]
                for f_new in range(n_fac):
                    if f_new == f_old:
                        continue
                    if current_load[f_new] + demands[i] > capacities[f_new] + 1e-9:
                        continue
                    
                    # Compute delta
                    delta = assign_cost[i, f_new] - assign_cost[i, f_old]
                    if current_load[f_new] == 0:
                        delta += open_cost[f_new]
                    if current_load[f_old] == demands[i]:
                        delta -= open_cost[f_old]
                    
                    if delta < -1e-9:
                        # Accept improving move
                        current_load[f_old] -= demands[i]
                        current_load[f_new] += demands[i]
                        current_res[i] = f_new
                        current_score += delta
                        improved = True
                        
                        if current_score < best_score:
                            best_score = current_score
                            best_res = current_res.copy()
                            best_load = current_load.copy()
                        
                        break  # Only do one move per customer per iteration
            
            # Try swap moves
            for i in range(n_cust):
                for j in range(i + 1, n_cust):
                    f_i = current_res[i]
                    f_j = current_res[j]
                    
                    if f_i == f_j:
                        continue
                    
                    # Check capacity
                    if current_load[f_i] - demands[i] + demands[j] > capacities[f_i] + 1e-9:
                        continue
                    if current_load[f_j] - demands[j] + demands[i] > capacities[f_j] + 1e-9:
                        continue
                    
                    # Compute delta
                    delta = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
                    
                    if delta < -1e-9:
                        # Accept improving move
                        current_load[f_i] += demands[j] - demands[i]
                        current_load[f_j] += demands[i] - demands[j]
                        current_res[i], current_res[j] = f_j, f_i
                        current_score += delta
                        improved = True
                        
                        if current_score < best_score:
                            best_score = current_score
                            best_res = current_res.copy()
                            best_load = current_load.copy()
        
        return best_res, best_load, best_score
    
    def greedy_init(seed):
        """Greedy initialization with randomization."""
        random.seed(seed)
        res = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        
        order = np.argsort(-demands)
        
        for i in order:
            fac_costs = assign_cost[i]
            noisy_costs = fac_costs + np.random.uniform(0, 1, n_fac) * np.std(fac_costs)
            fac_order_noisy = np.argsort(noisy_costs)
            
            assigned = False
            for f in fac_order_noisy:
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    res[i] = f
                    load[f] += demands[i]
                    assigned = True
                    break
            
            if not assigned:
                space = capacities - load
                max_space = np.max(space)
                candidates = np.where(space == max_space)[0]
                f = random.choice(candidates.tolist())
                res[i] = f
                load[f] += demands[i]
                
        return res
    
    best_assign = None
    best_score = np.inf
    
    # Phase 1: Lagrangian relaxation
    lr_assign, lr_score = lagrangian_relaxation(max_iter=150)
    if lr_assign is not None and lr_score < best_score:
        best_score = lr_score
        best_assign = lr_assign.copy()
    
    # Phase 2: Local search on Lagrangian solution
    if best_assign is not None and time.time() < deadline - 20.0:
        load = np.zeros(n_fac)
        np.add.at(load, best_assign, demands)
        ls_assign, ls_load, ls_score = local_search(best_assign, load, max_iter=2000)
        if ls_score < best_score:
            best_score = ls_score
            best_assign = ls_assign.copy()
    
    # Phase 3: Multiple greedy initializations with local search
    for r in range(3):
        if time.time() > deadline - 10.0:
            break
        
        res = greedy_init(r + 100)
        load = np.zeros(n_fac)
        np.add.at(load, res, demands)
        
        ls_assign, ls_load, ls_score = local_search(res, load, max_iter=1000)
        if ls_score < best_score:
            best_score = ls_score
            best_assign = ls_assign.copy()
    
    # Final: ensure we have a valid solution
    if best_assign is None:
        # Fallback to simple greedy
        best_assign = greedy_init(0)
    
    return best_assign.astype(int)
