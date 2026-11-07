import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit

    n_customers = len(demands)
    n_facilities = len(capacities)

    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    best_result = None
    best_score = float('inf')

    def evaluate(result):
        load = np.zeros(n_facilities)
        # Vectorized load calculation
        np.add.at(load, result, demands)
        
        if np.any(load > capacities + 1e-9):
            return float('inf')
        
        open_mask = np.zeros(n_facilities, dtype=bool)
        open_mask[result] = True
        
        open_sum = np.sum(open_cost[open_mask])
        assign_sum = np.sum(assign_cost[np.arange(n_customers), result])
        return open_sum + assign_sum

    def greedy_init(randomize=False, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        is_open = np.zeros(n_facilities, dtype=bool)
        
        # Sort customers by demand descending
        order = np.argsort(-demands)
        
        for i in order:
            d = demands[i]
            # Feasible facilities
            feasible = load + d <= capacities
            
            if not np.any(feasible):
                # Fallback: pick cheapest
                f = np.argmin(assign_cost[i])
                result[i] = f
                load[f] += d
                is_open[f] = True
                continue
            
            # Calculate marginal costs
            marginal_costs = assign_cost[i].copy()
            closed_mask = ~is_open
            marginal_costs[closed_mask] += open_cost[closed_mask]
            
            # Mask infeasible
            marginal_costs[~feasible] = np.inf
            
            if randomize:
                # Pick from top 3
                sorted_indices = np.argsort(marginal_costs)
                top_k = sorted_indices[:min(3, n_facilities)]
                f = top_k[np.random.randint(len(top_k))]
            else:
                f = np.argmin(marginal_costs)
            
            result[i] = f
            load[f] += d
            is_open[f] = True
            
        return result, load, is_open

    def fast_local_search(result, load, is_open):
        """
        Vectorized 1-move local search.
        Iterates until no improvement is found or time runs out.
        """
        improved = True
        # We use a 'best improvement' strategy per pass, but apply moves one by one 
        # to maintain consistency of load/is_open states.
        # To speed up, we can calculate the best move for ALL customers at once 
        # based on the current state, then apply them.
        
        while improved:
            improved = False
            # Calculate the best potential move for every customer simultaneously
            # Cost matrix: (n_customers, n_facilities)
            # current_cost: (n_customers,)
            
            # 1. Base assignment costs
            # For each customer i, cost to assign to f is assign_cost[i, f]
            
            # 2. Adjust for open costs
            # If facility f is closed, adding open_cost[f]
            # If moving customer i from f_old makes f_old empty, subtract open_cost[f_old]
            
            # Let's compute the delta for moving i to f.
            # delta(i, f) = assign_cost[i, f] - assign_cost[i, f_old] 
            #              + (open_cost[f] if not is_open[f] else 0)
            #              - (open_cost[f_old] if load[f_old] == demand[i] else 0)
            
            # This is hard to fully vectorize in one shot because 'load[f_old] == demand[i]' 
            # depends on the specific customer i and its current facility f_old.
            
            # Instead, we iterate customers, but vectorize the facility selection.
            # This is O(N * F) but the F part is fast NumPy.
            
            for i in range(n_customers):
                if time.time() > deadline - 5:
                    return result, load, is_open
                
                f_old = result[i]
                d = demands[i]
                
                # Calculate cost to move to any facility f
                # Start with assign_cost[i, :]
                costs = assign_cost[i].copy()
                
                # Add open cost for closed facilities
                costs[~is_open] += open_cost[~is_open]
                
                # If moving this customer empties f_old, we save open_cost[f_old]
                # This is a constant discount for all target facilities f != f_old
                if load[f_old] == d:
                    # We will save open_cost[f_old] if we move away
                    # So subtract it from all costs except f_old (where we stay)
                    # Actually, if we stay, we don't save it. If we move, we save it.
                    # So for f != f_old, cost -= open_cost[f_old]
                    costs[:] -= open_cost[f_old]
                    # But for f_old, we didn't actually move, so we shouldn't have subtracted it
                    # If we choose f_old, the delta is 0.
                    # Let's handle f_old separately or just mask it.
                    
                # Mask infeasible facilities
                feasible = load + d <= capacities
                costs[~feasible] = np.inf
                
                # Exclude current facility? 
                # If we stay, delta is 0. 
                # If we move, delta is costs[f] - (current_cost + adjustments).
                # It's easier to just find the min cost and compare to current effective cost.
                
                # Current effective cost for i:
                # assign_cost[i, f_old] 
                # + (open_cost[f_old] if is_open[f_old] and load[f_old]==d else 0) -> No, open cost is shared.
                # The "marginal" cost approach is safer.
                
                # Let's stick to the delta calculation which is more robust.
                # Delta(i, f) = (assign_cost[i, f] - assign_cost[i, f_old]) 
                #              + (open_cost[f] if not is_open[f] else 0)
                #              - (open_cost[f_old] if load[f_old] == d else 0)
                
                delta_costs = assign_cost[i] - assign_cost[i, f_old]
                
                # Add open cost for closed targets
                closed_mask = ~is_open
                delta_costs[closed_mask] += open_cost[closed_mask]
                
                # Subtract saved open cost if source becomes empty
                if load[f_old] == d:
                    # If we move to any f != f_old, we save open_cost[f_old]
                    # So subtract open_cost[f_old] from all f != f_old
                    delta_costs[:] -= open_cost[f_old]
                    # If we stay at f_old, we don't save it. 
                    # But delta_costs[f_old] is 0 (assign diff) + 0 (open diff, since it stays open) 
                    # Wait, if load[f_old]==d, and we stay, it stays open. 
                    # If we move, it closes. 
                    # So for f != f_old, delta includes -open_cost[f_old].
                    # For f == f_old, delta is 0.
                    # My calculation: delta_costs[f_old] = 0 - open_cost[f_old] (from the global subtract).
                    # This is incorrect for f_old.
                    # Let's fix:
                    delta_costs[f_old] = 0.0 # Staying is always delta 0
            
                # Mask infeasible
                feasible = load + d <= capacities
                delta_costs[~feasible] = np.inf
                
                # Find best move
                best_f_new = np.argmin(delta_costs)
                best_delta = delta_costs[best_f_new]
                
                # Only apply if strictly improving (and not staying)
                if best_f_new != f_old and best_delta < -1e-9:
                    # Apply move
                    load[f_old] -= d
                    if load[f_old] == 0:
                        is_open[f_old] = False
                    
                    load[best_f_new] += d
                    if load[best_f_new] == d: # Was 0
                        is_open[best_f_new] = True
                    
                    result[i] = best_f_new
                    improved = True
                    
        return result, load, is_open

    def swap_search(result, load, is_open):
        """
        2-swap local search.
        """
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 10:
                return result, load, is_open
                
            for i in range(n_customers):
                if time.time() > deadline - 10:
                    return result, load, is_open
                f_i = result[i]
                d_i = demands[i]
                
                for j in range(i + 1, n_customers):
                    f_j = result[j]
                    d_j = demands[j]
                    
                    if f_i == f_j:
                        continue
                    
                    # Check capacity
                    if load[f_i] - d_i + d_j > capacities[f_i]:
                        continue
                    if load[f_j] - d_j + d_i > capacities[f_j]:
                        continue
                    
                    # Calculate delta
                    delta_assign = (assign_cost[i, f_j] - assign_cost[i, f_i]) + \
                                   (assign_cost[j, f_i] - assign_cost[j, f_j])
                    
                    # Open cost delta
                    new_load_i = load[f_i] - d_i + d_j
                    new_load_j = load[f_j] - d_j + d_i
                    
                    delta_open = 0.0
                    # Facility i
                    if is_open[f_i]:
                        if new_load_i == 0:
                            delta_open -= open_cost[f_i]
                    else:
                        if new_load_i > 0:
                            delta_open += open_cost[f_i]
                            
                    # Facility j
                    if is_open[f_j]:
                        if new_load_j == 0:
                            delta_open -= open_cost[f_j]
                    else:
                        if new_load_j > 0:
                            delta_open += open_cost[f_j]
                    
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < -1e-9:
                        # Apply swap
                        result[i] = f_j
                        result[j] = f_i
                        
                        load[f_i] = new_load_i
                        load[f_j] = new_load_j
                        
                        if new_load_i == 0:
                            is_open[f_i] = False
                        else:
                            is_open[f_i] = True
                            
                        if new_load_j == 0:
                            is_open[f_j] = False
                        else:
                            is_open[f_j] = True
                        
                        improved = True
                        break # Restart inner loop after a change? Or continue? 
                        # Breaking inner loop and continuing outer is standard to avoid using stale loads
                        # for the same i with other j's.
        
        return result, load, is_open

    def solve_milp():
        """Solve the CFLP as a MILP using SciPy's HiGHS solver."""
        # Variables:
        # x[i, f] = 1 if customer i assigned to facility f (n_customers * n_facilities)
        # y[f] = 1 if facility f is open (n_facilities)
        # Total variables: n_customers * n_facilities + n_facilities
        
        n_vars = n_customers * n_facilities + n_facilities
        
        # Objective: minimize sum of open costs + assignment costs
        c = np.zeros(n_vars)
        for i in range(n_customers):
            for f in range(n_facilities):
                c[i * n_facilities + f] = assign_cost[i, f]
        for f in range(n_facilities):
            c[n_customers * n_facilities + f] = open_cost[f]
        
        # Constraints:
        # 1. Each customer assigned to exactly one facility: sum_f x[i,f] = 1 for all i
        # 2. Capacity: sum_i demands[i] * x[i,f] <= capacities[f] * y[f] for all f
        # 3. x[i,f] <= y[f] for all i,f (if customer assigned, facility must be open)
        
        A_rows = []
        lb = []
        ub = []
        
        # Constraint 1: assignment constraints
        for i in range(n_customers):
            row = np.zeros(n_vars)
            for f in range(n_facilities):
                row[i * n_facilities + f] = 1.0
            A_rows.append(row)
            lb.append(1.0)
            ub.append(1.0)
        
        # Constraint 2: capacity constraints (linearized)
        # sum_i demands[i] * x[i,f] - capacities[f] * y[f] <= 0
        for f in range(n_facilities):
            row = np.zeros(n_vars)
            for i in range(n_customers):
                row[i * n_facilities + f] = demands[i]
            row[n_customers * n_facilities + f] = -capacities[f]
            A_rows.append(row)
            lb.append(-np.inf)
            ub.append(0.0)
        
        # Constraint 3: x[i,f] <= y[f]
        for i in range(n_customers):
            for f in range(n_facilities):
                row = np.zeros(n_vars)
                row[i * n_facilities + f] = 1.0
                row[n_customers * n_facilities + f] = -1.0
                A_rows.append(row)
                lb.append(-np.inf)
                ub.append(0.0)
        
        A = np.array(A_rows)
        
        # Integer constraints: all variables are binary
        integrality = np.ones(n_vars)
        
        # Bounds: all variables in [0, 1]
        bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
        
        try:
            res_milp = milp(
                c=c,
                constraints=LinearConstraint(A, lb, ub),
                integrality=integrality,
                bounds=bounds
            )
            
            if res_milp.status == 0 and res_milp.x is not None:
                x = res_milp.x[:n_customers * n_facilities].reshape(n_customers, n_facilities)
                assignment = np.argmax(x, axis=1).astype(int)
                
                # Verify feasibility
                load_check = np.zeros(n_facilities)
                np.add.at(load_check, assignment, demands)
                if np.any(load_check > capacities + 1e-6):
                    return None
                
                return assignment
            else:
                return None
        except Exception:
            return None

    # Try MILP first
    if time.time() < deadline - 30:
        milp_result = solve_milp()
        if milp_result is not None:
            score = evaluate(milp_result)
            if score < best_score:
                best_score = score
                best_result = milp_result.copy()
    
    # Main loop - heuristic search
    # 1. Deterministic start
    res, load, is_open = greedy_init(randomize=False)
    res, load, is_open = fast_local_search(res, load, is_open)
    res, load, is_open = swap_search(res, load, is_open)
    score = evaluate(res)
    if score < best_score:
        best_score = score
        best_result = res.copy()

    # 2. Randomized restarts
    max_restarts = 200 
    for k in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        res, load, is_open = greedy_init(randomize=True, seed=k)
        res, load, is_open = fast_local_search(res, load, is_open)
        res, load, is_open = swap_search(res, load, is_open)
        score = evaluate(res)
        if score < best_score:
            best_score = score
            best_result = res.copy()

    return best_result
