import numpy as np
import time
from scipy.optimize import linprog

def assign(demands, capacities, open_cost, assign_cost):
    N = len(demands)
    M = len(capacities)
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit

    def calc_cost(sol, loads):
        open_mask = loads > 0
        cost = np.sum(open_cost[open_mask])
        assign_sum = np.sum(assign_cost[np.arange(N), sol])
        return cost + assign_sum

    # 1. Greedy Initialization
    order = np.argsort(-demands)
    current_solution = np.zeros(N, dtype=int)
    load = np.zeros(M)
    
    for i in order:
        feasible_mask = load + demands[i] <= capacities
        if not np.any(feasible_mask):
            idx = np.argmax(capacities - load)
            current_solution[i] = idx
            load[idx] += demands[i]
            continue
            
        costs = assign_cost[i].copy()
        costs[~feasible_mask] = np.inf
        f = np.argmin(costs)
        current_solution[i] = f
        load[f] += demands[i]
        
    current_cost = calc_cost(current_solution, load)
    best_solution = current_solution.copy()
    best_cost = current_cost

    # 2. Solve LP Relaxation using a smarter formulation if time permits
    # We use the standard CFLP relaxation.
    # Min sum_{i,j} c_ij x_ij + sum_j F_j y_j
    # s.t. sum_j x_ij = d_i
    #      sum_i x_ij <= cap_j
    #      x_ij <= d_i y_j
    #      0 <= x_ij <= d_i, 0 <= y_j <= 1
    
    def solve_lp():
        n_vars = N * M + M
        c_obj = np.zeros(n_vars)
        c_obj[:N*M] = assign_cost.flatten()
        c_obj[N*M:] = open_cost
        
        # Equality constraints: sum_j x_ij = d_i
        A_eq = np.zeros((N, n_vars))
        b_eq = demands.copy()
        for i in range(N):
            A_eq[i, i*M:(i+1)*M] = 1.0
            
        # Inequality constraints:
        # 1. sum_i x_ij <= cap_j  (M constraints)
        # 2. x_ij - d_i y_j <= 0  (N*M constraints)
        A_ub = np.zeros((M + N*M, n_vars))
        b_ub = np.zeros(M + N*M)
        
        # Capacity constraints
        for j in range(M):
            A_ub[j, j::M] = 1.0 # This indexing is wrong for flattened x. 
            # x is flattened as i*M+j. So column index for x[i,j] is i*M+j.
            # For a fixed j, the columns are j, M+j, 2M+j, ...
            # Let's construct this properly.
            cols = np.array([i*M + j for i in range(N)])
            A_ub[j, cols] = 1.0
            b_ub[j] = capacities[j]
            
        # Linking constraints
        row_idx = M
        for i in range(N):
            for j in range(M):
                col_x = i*M + j
                col_y = N*M + j
                A_ub[row_idx, col_x] = 1.0
                A_ub[row_idx, col_y] = -demands[i]
                b_ub[row_idx] = 0.0
                row_idx += 1
                
        bounds = [(0, demands[i]) for i in range(N*M)] + [(0, 1) for _ in range(M)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                         bounds=bounds, method='highs', options={"presolve": True})
            if res.status != 0:
                return None
            x = res.x[:N*M].reshape(N, M)
            # We don't strictly need y for rounding, but it helps prioritize
            y = res.x[N*M:]
            return x, y
        except Exception:
            return None

    if time.time() - start_time < deadline - 20:
        lp_res = solve_lp()
        if lp_res is not None:
            x_lp, y_lp = lp_res
            
            # Round the LP solution
            # Strategy: Sort customers by demand descending.
            # For each customer, pick the facility with highest LP flow, 
            # breaking ties by cost, subject to capacity.
            sol_lp = np.zeros(N, dtype=int)
            loads_lp = np.zeros(M)
            
            # Sort customers by demand descending
            order_lp = np.argsort(-demands)
            
            for i in order_lp:
                feasible = loads_lp + demands[i] <= capacities
                if not np.any(feasible):
                    feasible = np.ones(M, dtype=bool)
                
                # Score: high LP flow, low cost
                # We want to maximize flow, so use -flow as cost?
                # Or just pick max flow among feasible.
                # Let's use a combined score.
                scores = np.full(M, -np.inf)
                for j in range(M):
                    if feasible[j]:
                        # Prioritize LP flow, then low cost
                        scores[j] = x_lp[i, j] * 1000.0 - assign_cost[i, j]
                
                best_j = np.argmax(scores)
                if scores[best_j] == -np.inf:
                    # Fallback to cost
                    costs = assign_cost[i].copy()
                    costs[~feasible] = np.inf
                    best_j = np.argmin(costs)
                    
                sol_lp[i] = best_j
                loads_lp[best_j] += demands[i]
                
            cost_lp = calc_cost(sol_lp, loads_lp)
            if cost_lp < best_cost:
                best_cost = cost_lp
                best_solution = sol_lp.copy()
                current_solution = sol_lp.copy()
                load = loads_lp.copy()
                current_cost = cost_lp

    # 3. Local Search
    # Use a more efficient local search with time checks
    while time.time() - start_time < deadline:
        improved = False
        
        # Single move
        best_delta = 0
        best_move = None
        
        # Vectorize the cost difference calculation?
        # d_assign = assign_cost[i, k] - assign_cost[i, curr_f]
        # We can precompute matrices.
        
        for i in range(N):
            curr_f = current_solution[i]
            # Check all k
            feasible_k = load + demands[i] <= capacities
            feasible_k[curr_f] = False
            
            if not np.any(feasible_k):
                continue
                
            d_assign = assign_cost[i, feasible_k] - assign_cost[i, curr_f]
            
            # d_open calculation
            d_open = np.zeros(M)
            new_load_curr = load[curr_f] - demands[i]
            if new_load_curr == 0:
                d_open[curr_f] -= open_cost[curr_f]
            
            # For each feasible k, if load[k] == 0, add open_cost[k]
            zero_loads = load == 0
            d_open[zero_loads] += open_cost[zero_loads]
            
            total_delta = d_assign + d_open[feasible_k]
            
            min_idx = np.argmin(total_delta)
            if total_delta[min_idx] < best_delta:
                best_delta = total_delta[min_idx]
                best_move = (i, np.where(feasible_k)[0][min_idx])
                
        if best_move is not None:
            i, k = best_move
            curr_f = current_solution[i]
            load[curr_f] -= demands[i]
            load[k] += demands[i]
            current_solution[i] = k
            current_cost += best_delta
            improved = True
            if current_cost < best_cost:
                best_cost = current_cost
                best_solution = current_solution.copy()
                
        # Swap
        best_delta = 0
        best_swap = None
        
        for i in range(N):
            f_i = current_solution[i]
            for m in range(i + 1, N):
                f_m = current_solution[m]
                if f_i == f_m:
                    continue
                
                new_load_fi = load[f_i] - demands[i] + demands[m]
                new_load_fm = load[f_m] - demands[m] + demands[i]
                
                if new_load_fi > capacities[f_i] or new_load_fm > capacities[f_m]:
                    continue
                
                d_assign = (assign_cost[i, f_m] + assign_cost[m, f_i]) - (assign_cost[i, f_i] + assign_cost[m, f_m])
                
                d_open = 0
                if new_load_fi == 0:
                    d_open -= open_cost[f_i]
                if new_load_fm == 0:
                    d_open -= open_cost[f_m]
                
                total_delta = d_assign + d_open
                
                if total_delta < best_delta:
                    best_delta = total_delta
                    best_swap = (i, m)
                    
        if best_swap is not None:
            i, m = best_swap
            f_i = current_solution[i]
            f_m = current_solution[m]
            
            load[f_i] -= demands[i]
            load[f_i] += demands[m]
            load[f_m] -= demands[m]
            load[f_m] += demands[i]
            
            current_solution[i] = f_m
            current_solution[m] = f_i
            
            current_cost += best_delta
            improved = True
            if current_cost < best_cost:
                best_cost = current_cost
                best_solution = current_solution.copy()
                
        if not improved:
            break
            
    return best_solution.astype(int)
