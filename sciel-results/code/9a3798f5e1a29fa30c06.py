import numpy as np
import time
from scipy.optimize import linprog

def heuristics(prizes, weights):
    # 1. Input processing and validation
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Ensure shapes are (5, 100)
    if prizes.ndim == 1:
        # Fallback if 1D, though contract says (5, 100)
        P = prizes
        w = weights.reshape(5, -1) if weights.size == prizes.size * 5 else weights
    else:
        # prizes is (5, 100), weights is (5, 100)
        # "Total prize" per item is the sum across the 5 dimensions
        P = prizes.sum(axis=0) # (100,)
        w = weights             # (5, 100)
        
    n = P.shape[0]
    m = w.shape[0]
    capacity = 25.0
    
    # Time Guard
    t_start = time.time()
    deadline = t_start + 290.0  # 10s safety margin
    
    def check_time():
        return time.time() > deadline
        
    # 2. Solve LP Relaxation to get Dual Prices (Shadow Prices)
    # Problem:
    # Maximize: P^T x
    # Subject to: W x <= c * 1 (where c=25)
    #             0 <= x <= 1
    #
    # scipy.optimize.linprog minimizes c^T x subject to A_ub x <= b_ub
    # So we minimize -P^T x
    # A_ub is the weight matrix w (5, 100)
    # b_ub is [25, 25, 25, 25, 25]
    
    c_obj = -P
    A_ub = w
    b_ub = np.full(m, capacity)
    bounds = [(0, 1) for _ in range(n)]
    
    try:
        # Method 'highs' is fast and robust for this size
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.status == 0: # Optimal
            # res.ineqlin.marginals are the dual variables (shadow prices) for the inequality constraints
            # Since we minimized -P, the duals correspond to the constraints W x <= b.
            # Note: linprog returns marginals as the negative of the standard dual variables 
            # depending on the sign convention, but for minimization with <= constraints,
            # the marginals are non-negative and represent the increase in objective per unit increase in RHS.
            # Wait, standard LP duality:
            # Min c^T x s.t. Ax <= b. Dual: Max b^T y s.t. A^T y >= c, y >= 0.
            # scipy returns marginals for constraints. For 'highs', res.ineqlin.marginals 
            # gives the dual values.
            
            duals = res.ineqlin.marginals
            # Ensure duals are non-negative (they should be for <= constraints in min problem)
            duals = np.maximum(duals, 0.0)
            
            # Calculate Reduced Costs
            # Reduced cost for item j = c_j - sum_i dual_i * a_ij
            # Here c_j is the prize P[j] (since we want to maximize P, in min form it's -P, 
            # but the "efficiency" logic holds: we want items where Prize > Cost of Resources)
            # The duals tell us the "value" of consuming 1 unit of constraint i.
            # If P[j] > sum(duals * weights[j]), the item is "profitable" beyond the marginal cost.
            # Actually, the reduced cost in the dual sense for the primal variable x_j is:
            # rc_j = P_j - (W^T duals)_j
            # If rc_j > 0, the item is "attractive" because it provides more prize than the 
            # marginal value of the resources it consumes.
            
            resource_cost = w.T @ duals # (100,)
            reduced_costs = P - resource_cost
            
            # Heuristic: Items with higher reduced cost are more attractive.
            # However, reduced costs can be negative.
            # We need positive values.
            # We can shift them or use a softplus-like transformation.
            # A common trick: heuristic = max(reduced_cost, epsilon)
            # Or better: rank-based or exponential to emphasize top items.
            # Let's use: h = exp(k * reduced_cost) or just clip to positive.
            # Given the ACO uses pheromone * heuristic, and pheromone starts at 1,
            # large values help.
            
            # Normalize reduced costs to be positive and comparable
            # If all are negative, it means the LP solution is at the boundary and no item is "free".
            # In that case, we fall back to prize/weight ratio.
            
            if np.any(reduced_costs > 0):
                # Shift so min is small positive
                min_rc = np.min(reduced_costs)
                shifted_rc = reduced_costs - min_rc
                # Scale to avoid overflow in exp if we use it, but simple positive is safer
                # Let's use a power law to boost high reduced costs
                h = shifted_rc + 1e-6
            else:
                # Fallback: Simple prize/weight ratio
                wsum = w.sum(axis=0)
                h = P / (wsum + 1e-6)
                
        else:
            # Fallback if LP fails
            wsum = w.sum(axis=0)
            h = P / (wsum + 1e-6)
            
    except Exception:
        # Fallback if LP fails
        wsum = w.sum(axis=0)
        h = P / (wsum + 1e-6)
        
    # 3. Sanitize Heuristic
    # Ensure positive and finite
    h = np.asarray(h, dtype=float)
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    h = np.clip(h, 1e-9, 1e6)
    
    # Optional: Quick sanity check simulation?
    # The LP dual approach is theoretically strong. 
    # To be safe, we can run a very fast greedy check to see if this heuristic 
    # yields a good total prize compared to a simple ratio.
    # But the LP duals are derived from the optimal continuous solution, 
    # which is a very strong guide for the discrete knapsack.
    
    if check_time():
         return h

    # 4. (Optional) Refinement via Fast Local Search on Heuristic Scaling
    # Sometimes the raw reduced costs might be too flat or too peaked.
    # We can try a few scaling factors k: h_final = h ** k
    # And pick the one that gives the best greedy score.
    # This is much faster than full ACO simulation.
    
    def greedy_score(heuristic, P, w, n, m, cap):
        rem = np.full(m, cap)
        used = np.zeros(n, dtype=bool)
        total = 0.0
        # Sort by heuristic descending
        order = np.argsort(-heuristic)
        for idx in order:
            if used[idx]:
                continue
            if np.all(w[:, idx] <= rem):
                used[idx] = True
                rem -= w[:, idx]
                total += P[idx]
        return total
    
    best_h = h
    best_score = greedy_score(h, P, w, n, m, capacity)
    
    # Try a few powers to sharpen or smooth the preference
    for power in [0.5, 2.0, 3.0]:
        if check_time():
            break
        h_test = h ** power
        h_test = np.where(np.isfinite(h_test) & (h_test > 0), h_test, 1e-9)
        h_test = np.clip(h_test, 1e-9, 1e6)
        
        score = greedy_score(h_test, P, w, n, m, capacity)
        if score > best_score:
            best_score = score
            best_h = h_test
            
    return best_h
