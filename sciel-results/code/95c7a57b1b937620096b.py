"""Capacitated facility location: Frank-Wolfe continuous relaxation + Greedy Projection + Local Search."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def assign(demands, capacities, open_cost, assign_cost):
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Safety margin: 150s limit, aim for 140s max
    deadline = time.time() + 140.0
    
    # 1. Frank-Wolfe (Conditional Gradient) for the continuous relaxation
    # Objective: minimize sum_{i,j} c_ij x_ij + sum_j f_j max(0, sum_i d_i x_ij)
    # Variables x_ij >= 0, sum_j x_ij = 1 for each i.
    # This is a convex optimization problem.
    
    # We represent the solution as a matrix X of shape (n_cust, n_fac)
    # where X[i, j] is the fraction of customer i assigned to facility j.
    
    # Initialize with a random feasible point (uniform distribution over facilities)
    # Or better, initialize with the greedy assignment projected to 0/1 then smoothed?
    # Let's start with uniform 1/n_fac for simplicity and robustness.
    X = np.ones((n_cust, n_fac)) / n_fac
    
    # Precompute demand vector
    d = np.array(demands)
    
    # Helper to compute gradient of the objective
    def compute_gradient(X):
        # Grad w.r.t X[i, j]
        # Term 1: c_ij
        # Term 2: f_j * indicator(sum_i d_i X[i, j] > 0) * d_i
        # Since we are in the interior or boundary, the derivative of max(0, load) is 1 if load > 0.
        
        # Load at each facility
        load = X.T @ d  # shape (n_fac,)
        
        # Gradient from assignment cost
        grad_assign = np.array(assign_cost)  # shape (n_cust, n_fac)
        
        # Gradient from fixed cost
        # If load[j] > 0, the marginal cost of adding d_i to j is f_j.
        # Note: The constraint is sum_j X[i, j] = 1.
        # The Lagrangian includes multipliers for the equality constraints.
        # However, Frank-Wolfe usually works with the gradient of the function f(X)
        # subject to constraints. The linearization step minimizes <grad, Y - X> subject to Y in P.
        # Here P is the product of simplices.
        
        # The effective gradient for the linear minimization step is:
        # G[i, j] = c_ij + f_j * (1 if load[j] > 0 else 0)
        # Wait, the fixed cost is f_j * max(0, load_j).
        # If load_j > 0, derivative is f_j. If load_j == 0, derivative is 0.
        # So the term is f_j * I(load_j > 0).
        
        active = load > 1e-9
        grad_fixed = np.outer(np.ones(n_cust), open_cost * active)
        
        return grad_assign + grad_fixed
    
    # Frank-Wolfe iterations
    # We want to minimize F(X) = sum c_ij X_ij + sum f_j max(0, sum d_i X_ij)
    
    # The linear minimization oracle for the product of simplices:
    # Minimize sum_{i,j} G[i, j] Y[i, j] subject to sum_j Y[i, j] = 1, Y[i, j] >= 0.
    # This decouples for each i. For each i, we pick j that minimizes G[i, j].
    
    # Step size rule: Standard FW uses alpha_t = 2/(t+2) or backtracking.
    # For convex problems, alpha_t = 2/(t+2) is standard.
    
    max_iter_fw = 2000
    X_curr = X.copy()
    
    for t in range(max_iter_fw):
        if time.time() > deadline - 20.0:
            break
            
        G = compute_gradient(X_curr)
        
        # Solve linear minimization oracle
        # For each customer i, find argmin_j G[i, j]
        # Y is a 0-1 matrix (vertices of the simplex)
        Y_indices = np.argmin(G, axis=1)
        Y = np.zeros_like(X_curr)
        Y[np.arange(n_cust), Y_indices] = 1.0
        
        # Step size
        # alpha = 2.0 / (t + 2.0)
        # Sometimes backtracking is better, but standard is fine for this scale.
        alpha = 2.0 / (t + 2.0)
        
        X_new = (1 - alpha) * X_curr + alpha * Y
        
        # Check convergence
        # If X_curr is close to Y, we are at a vertex.
        if np.allclose(X_curr, Y, atol=1e-6):
            break
            
        X_curr = X_new
        
    # 2. Projection: Convert fractional X to integer assignment
    # We need to assign each customer to exactly one facility, respecting capacities.
    # This is a transportation problem.
    # We can solve it using a greedy approach or a min-cost flow.
    # Given the size (100x20), a simple greedy with repair or a linear program solver is fast.
    # Since we don't have a dedicated LP solver in stdlib, we use a greedy heuristic 
    # based on the fractional solution as a prior.
    
    # Sort customers by "confidence" or just process all.
    # A good heuristic:
    # 1. Compute "soft" assignment costs.
    # 2. Use a greedy assignment where we prioritize customers with high fractional weight on their best facility.
    
    # Let's use a more robust method: 
    # Min-cost flow formulation solved via successive shortest paths or just greedy if tight.
    # Actually, with 100 customers and 20 facilities, a simple greedy might fail if capacities are tight.
    # Let's use a "Greedy with Backtracking" or simply solve the assignment problem 
    # where cost is modified by the fractional solution?
    
    # Better approach: 
    # Use the fractional solution to guide a greedy assignment.
    # Sort customers by the margin of their top choice in X_curr.
    # Or, simply use the fractional solution to define a priority.
    
    # Let's try a direct greedy assignment using the costs, but initialized by the FW solution.
    # We can define a "virtual cost" c'_ij = c_ij - lambda * X_curr[i,j] ? No.
    
    # Let's use a standard greedy:
    # 1. Start with empty facilities.
    # 2. For each customer (ordered by demand descending for packing efficiency, 
    #    or by fractional confidence), assign to the facility with lowest 
    #    (assign_cost + open_cost_indicator) that has capacity.
    
    # To leverage FW, we can bias the initial choice.
    # Let's create a list of (customer, best_facility_in_FW) pairs.
    
    # Actually, the FW solution gives us a good "centroid".
    # Let's project by solving a transportation problem:
    # Min sum c_ij z_ij
    # s.t. sum_j z_ij = 1
    #      sum_i d_i z_ij <= cap_j
    #      z_ij in {0,1}
    
    # Since we don't have a MILP solver, we approximate.
    # Greedy Strategy:
    # 1. Calculate total demand and total capacity.
    # 2. Sort facilities by open_cost / capacity (efficiency) or just open_cost.
    # 3. We will assign customers one by one.
    
    # Let's use a heuristic that respects the FW structure:
    # For each customer i, the FW solution X_curr[i, :] suggests a distribution.
    # We want to pick j such that X_curr[i, j] is high AND c_ij is low.
    
    # Let's try a "Greedy with Capacities" where we sort customers by demand (largest first)
    # and for each, pick the feasible facility with the lowest effective cost.
    # Effective cost = c_ij + (open_cost[j] if load[j]==0 else 0)
    
    # To improve on pure greedy, we can use the FW solution to break ties or bias the cost.
    # modified_cost[i, j] = c_ij - beta * X_curr[i, j]
    # This encourages assignments that align with the continuous optimum.
    
    beta = 1.0  # Tuning parameter
    
    # We will perform multiple restarts of the greedy projection with different betas
    # and different orderings to find a good integer solution.
    
    best_int_res = None
    best_int_score = np.inf
    
    # Precompute for speed
    d_arr = np.array(demands)
    cap_arr = np.array(capacities)
    oc_arr = np.array(open_cost)
    ac_arr = np.array(assign_cost)
    
    def greedy_project(beta_val, order_type='demand'):
        res = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        
        if order_type == 'demand':
            order = np.argsort(-d_arr)
        elif order_type == 'fw_confidence':
            # Sort by how "decided" the FW solution is (max prob - second max prob)
            sorted_X = np.sort(X_curr, axis=1)
            confidence = sorted_X[:, -1] - sorted_X[:, -2]
            order = np.argsort(-confidence)
        else:
            order = np.arange(n_cust)
            
        for i in order:
            d_i = d_arr[i]
            # Feasible facilities
            feasible = np.where(load + d_i <= cap_arr)[0]
            
            if len(feasible) == 0:
                # Should not happen if total cap >= total demand
                # Fallback: pick max remaining capacity
                remaining = cap_arr - load
                f = np.argmax(remaining)
                res[i] = f
                load[f] += d_i
                continue
            
            # Calculate costs for feasible facilities
            # Base cost
            costs = ac_arr[i, feasible]
            
            # Open cost penalty if facility is currently empty
            # Note: This is a heuristic approximation of the marginal cost.
            # If load[feasible] == 0, we add open_cost.
            # Also add the FW bias.
            
            open_penalties = np.zeros(len(feasible))
            for idx, f in enumerate(feasible):
                if load[f] == 0:
                    open_penalties[idx] = oc_arr[f]
                    
            fw_bias = -beta_val * X_curr[i, feasible]
            
            total_cost = costs + open_penalties + fw_bias
            
            min_idx = np.argmin(total_cost)
            f_new = feasible[min_idx]
            
            res[i] = f_new
            load[f_new] += d_i
            
        return res, load
    
    def calc_cost(res):
        used = np.unique(res)
        return np.sum(oc_arr[used]) + np.sum(ac_arr[np.arange(n_cust), res])
    
    # Try several beta values and orderings
    betas = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    orderings = ['demand', 'fw_confidence']
    
    for b in betas:
        if time.time() > deadline - 10.0:
            break
        for o in orderings:
            if time.time() > deadline - 5.0:
                break
            res, load = greedy_project(b, o)
            if np.all(load <= cap_arr):
                score = calc_cost(res)
                if score < best_int_score:
                    best_int_score = score
                    best_int_res = res.copy()
                    
    # If no valid integer solution was found (unlikely), fall back to simple greedy
    if best_int_res is None:
        res, load = greedy_project(0.0, 'demand')
        best_int_res = res
        best_int_score = calc_cost(res)
        
    # 3. Local Search Polish
    # Use the best integer solution as a starting point for local search.
    # We reuse a simplified local search from the previous strategy to polish the solution.
    
    def local_search(res, load):
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 2.0:
                break
                
            # Single customer moves
            for i in range(n_cust):
                if time.time() > deadline - 2.0:
                    break
                    
                f_old = res[i]
                d_i = d_arr[i]
                
                feasible = (load + d_i <= cap_arr) & (np.arange(n_fac) != f_old)
                if not np.any(feasible):
                    continue
                
                delta_assign = ac_arr[i, feasible] - ac_arr[i, f_old]
                
                delta_open = np.zeros(np.sum(feasible))
                fac_indices = np.where(feasible)[0]
                
                if load[f_old] == d_i:
                    delta_open -= oc_arr[f_old]
                
                new_facs = fac_indices[load[fac_indices] == 0]
                if len(new_facs) > 0:
                    pos_in_fac = np.searchsorted(fac_indices, new_facs)
                    delta_open[pos_in_fac] += oc_arr[new_facs]
                
                total_delta = delta_assign + delta_open
                min_idx = np.argmin(total_delta)
                best_delta = total_delta[min_idx]
                
                if best_delta < -1e-9:
                    best_f_new = fac_indices[min_idx]
                    res[i] = best_f_new
                    load[f_old] -= d_i
                    load[best_f_new] += d_i
                    improved = True
            
            # Swap moves (limited sampling)
            n_swaps = 500
            for _ in range(n_swaps):
                if time.time() > deadline - 2.0:
                    break
                    
                i, j = np.random.randint(0, n_cust, size=2)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                    
                f_i = res[i]
                f_j = res[j]
                if f_i == f_j:
                    continue
                
                d_i = d_arr[i]
                d_j = d_arr[j]
                
                if load[f_j] - d_j + d_i > cap_arr[f_j]:
                    continue
                if load[f_i] - d_i + d_j > cap_arr[f_i]:
                    continue
                
                delta_assign = (ac_arr[i, f_j] - ac_arr[i, f_i]) + \
                               (ac_arr[j, f_i] - ac_arr[j, f_j])
                               
                # Open cost change is 0 for swaps (unless a facility becomes empty, 
                # which is impossible in a swap of two non-zero demand customers 
                # unless one of them was the *only* customer, but even then 
                # the other moves in, so load doesn't hit 0).
                # Wait: if load[f_i] == d_i, and we move i out and j in, load becomes d_j.
                # If d_j > 0, facility is still open. So open cost is unchanged.
                
                if delta_assign < -1e-9:
                    res[i], res[j] = res[j], res[i]
                    load[f_i] -= d_i
                    load[f_i] += d_j
                    load[f_j] -= d_j
                    load[f_j] += d_i
                    improved = True
                    
        return res, load
    
    current_res = best_int_res.copy()
    current_load = np.zeros(n_fac)
    for i in range(n_cust):
        current_load[current_res[i]] += d_arr[i]
        
    # Run local search
    current_res, current_load = local_search(current_res, current_load)
    final_score = calc_cost(current_res)
    
    if final_score < best_int_score:
        best_int_res = current_res
        best_int_score = final_score
        
    return best_int_res
# EVOLVE-BLOCK-END
