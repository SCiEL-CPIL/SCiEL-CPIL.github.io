import numpy as np
import time
import random
import scipy.optimize as so

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

    def solve_lp(res_init):
        """
        Solve the LP relaxation of the capacitated facility location problem
        initialized with res_init.
        Variables: x_j (open facility j), y_ij (assign i to j)
        Objective: sum(open_cost * x) + sum(assign_cost * y)
        Constraints:
        1. sum_j y_ij = 1 for all i
        2. sum_i demands_i * y_ij <= capacities_j * x_j for all j
        3. x_j <= 1
        4. x_j, y_ij >= 0
        """
        # Objective coefficients
        c = np.zeros(n_cust * n_fac + n_fac)
        c[:n_cust * n_fac] = assign_cost.flatten()
        c[n_cust * n_fac:] = open_cost
        
        # Inequality constraints (A_ub @ vars <= b_ub)
        # Constraint 2: sum_i d_i * y_ij - cap_j * x_j <= 0
        A_ub = np.zeros((n_fac, n_cust * n_fac + n_fac))
        b_ub = np.zeros(n_fac)
        
        for j in range(n_fac):
            # y_ij terms
            for i in range(n_cust):
                idx = i * n_fac + j
                A_ub[j, idx] = demands[i]
            # x_j term
            A_ub[j, n_cust * n_fac + j] = -capacities[j]
        
        # Equality constraints (A_eq @ vars = b_eq)
        # Constraint 1: sum_j y_ij = 1
        A_eq = np.zeros((n_cust, n_cust * n_fac + n_fac))
        b_eq = np.ones(n_cust)
        
        for i in range(n_cust):
            for j in range(n_fac):
                idx = i * n_fac + j
                A_eq[i, idx] = 1.0
        
        # Bounds
        bounds = [(0, 1)] * (n_cust * n_fac + n_fac)
        
        # Initial guess based on res_init
        x0 = np.zeros(n_cust * n_fac + n_fac)
        for i in range(n_cust):
            j = res_init[i]
            x0[i * n_fac + j] = 1.0
            x0[n_cust * n_fac + j] = 1.0
        
        # Solve
        try:
            res_opt = so.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs', x0=x0)
            if res_opt.status == 0:
                y = res_opt.x[:n_cust * n_fac].reshape(n_cust, n_fac)
                x = res_opt.x[n_cust * n_fac:]
                
                # Rounding: Assign each customer to the facility with highest y_ij
                res_rounded = np.argmax(y, axis=1)
                
                # Check feasibility of rounded solution
                load = np.zeros(n_fac)
                np.add.at(load, res_rounded, demands)
                if np.any(load > capacities + 1e-9):
                    return None, np.inf
                
                score = compute_score(res_rounded)
                return res_rounded, score
            else:
                return None, np.inf
        except Exception:
            return None, np.inf

    best_assign = None
    best_score = np.inf
    
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        res = greedy_init(r)
        load = np.zeros(n_fac)
        np.add.at(load, res, demands)
        
        # Local search refinement
        local_search(res, load, max_iter=100)
        
        score = compute_score(res)
        if score < best_score:
            best_score = score
            best_assign = res.copy()
            
        # Try LP relaxation with rounding
        if time.time() < deadline - 5.0:
            res_lp, score_lp = solve_lp(res)
            if res_lp is not None:
                if score_lp < best_score:
                    best_score = score_lp
                    best_assign = res_lp.copy()
                    
                # Refine LP solution with local search
                load_lp = np.zeros(n_fac)
                np.add.at(load_lp, res_lp, demands)
                local_search(res_lp, load_lp, max_iter=100)
                score_lp_refined = compute_score(res_lp)
                if score_lp_refined < best_score:
                    best_score = score_lp_refined
                    best_assign = res_lp.copy()

    return best_assign.astype(int)
