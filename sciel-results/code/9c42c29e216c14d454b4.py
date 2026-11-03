import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = 150.0 - 2.0  # 2s safety margin

    # 1. Initial Heuristic: Rank Matching
    # Facilities with high flow sums should go to locations with low distance sums.
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Sort facilities by descending flow sum
    heavy_order = np.argsort(-a_row_sums, kind="stable")
    # Sort locations by ascending distance sum
    central_order = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_order] = central_order
    
    def compute_cost(p):
        b_sub = b[np.ix_(p, p)]
        return int(np.sum(a * b_sub))

    current_cost = compute_cost(p)
    best_p = p.copy()
    best_cost = current_cost

    # Precompute for faster access
    # We need a[i], a[:,i], b[li], b[:,li]
    
    def get_delta(p, i, j):
        li, lj = p[i], p[j]
        # Vectorized delta calculation
        # delta = sum_k (a[i,k] - a[j,k]) * (b[lj, p[k]] - b[li, p[k]]) 
        #       + sum_k (a[k,i] - a[k,j]) * (b[p[k], lj] - b[p[k], li])
        
        # Note: a[i] is a view, a[:,i] is a view.
        # b[lj, p] is fancy indexing, creates a copy.
        # To avoid copies, we can use advanced indexing carefully or accept the cost.
        # For n=150, the cost of fancy indexing is low.
        
        diff_a_row = a[i] - a[j]
        diff_b_col = b[lj, p] - b[li, p]
        
        diff_a_col = a[:, i] - a[:, j]
        diff_b_row = b[p, lj] - b[p, li]
        
        return np.dot(diff_a_row, diff_b_col) + np.dot(diff_a_col, diff_b_row)

    # Local Search: First Fit Descent
    def local_search(p, current_cost, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(n):
                if time.time() > time_limit:
                    return p, current_cost
                for j in range(i + 1, n):
                    if time.time() > time_limit:
                        return p, current_cost
                    
                    delta = get_delta(p, i, j)
                    if delta < 0:
                        # Perform swap
                        p[i], p[j] = p[j], p[i]
                        current_cost += int(delta)
                        improved = True
                        # Restart the loop to check from the beginning
                        # This is standard First Fit behavior to ensure we don't miss better moves
                        # that become available due to the swap.
                        break
                if improved:
                    continue
                if time.time() > time_limit:
                    return p, current_cost
        return p, current_cost

    # Run initial local search
    p, current_cost = local_search(p, current_cost, deadline - 10)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()

    # Random Restarts
    # If we have time, perturb the best solution and try to improve it.
    # We use a fixed number of swaps for perturbation.
    while time.time() < deadline - 5:
        # Create a perturbed version of the best solution
        p_restart = best_p.copy()
        # Number of swaps: typically n/10 to n/5
        num_swaps = max(1, n // 10)
        for _ in range(num_swaps):
            i, j = np.random.choice(n, 2, replace=False)
            p_restart[i], p_restart[j] = p_restart[j], p_restart[i]
        
        # Calculate cost of perturbed solution
        # We can compute it directly or via deltas from best_p.
        # Direct computation is O(n^2) which is 22500 ops, very fast.
        current_cost_r = compute_cost(p_restart)
        
        # If the perturbation made it worse, we might skip local search if it's very bad?
        # No, local search can climb out. But if it's much worse, it might waste time.
        # However, for QAP, small perturbations are usually close.
        
        # Run local search on the perturbed solution
        # Allocate remaining time minus a small buffer
        time_left = deadline - 5 - time.time()
        if time_left < 1:
            break
            
        p_r, cost_r = local_search(p_restart, current_cost_r, time.time() + time_left)
        
        if cost_r < best_cost:
            best_cost = cost_r
            best_p = p_r.copy()
            
    return best_p.tolist()
