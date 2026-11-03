import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = 150.0 - 3.0  # 3s safety margin

    def compute_cost(p):
        b_sub = b[np.ix_(p, p)]
        return int(np.sum(a * b_sub))

    def get_delta(p, i, j):
        """Compute delta for swapping p[i] and p[j] using vectorized O(n) operations."""
        li, lj = p[i], p[j]
        # Vectorized delta:
        # delta = sum_k (a[i,k] - a[j,k]) * (b[lj, p[k]] - b[li, p[k]])
        #       + sum_k (a[k,i] - a[k,j]) * (b[p[k], lj] - b[p[k], li])
        diff_a_row = a[i] - a[j]
        diff_b_col = b[lj, p] - b[li, p]
        diff_a_col = a[:, i] - a[:, j]
        diff_b_row = b[p, lj] - b[p, li]
        return np.dot(diff_a_row, diff_b_col) + np.dot(diff_a_col, diff_b_row)

    def steepest_descent_local_search(p, current_cost, time_limit):
        """Steepest descent: find the best improving swap per pass."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_delta = 0
            best_i, best_j = -1, -1
            
            for i in range(n):
                if time.time() > time_limit:
                    return p, current_cost
                for j in range(i + 1, n):
                    delta = get_delta(p, i, j)
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
            
            if best_i != -1:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += int(best_delta)
                improved = True
        
        return p, current_cost

    def first_fit_local_search(p, current_cost, time_limit):
        """First fit: take the first improving swap found."""
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
                        p[i], p[j] = p[j], p[i]
                        current_cost += int(delta)
                        improved = True
                        break
                if improved:
                    continue
        return p, current_cost

    # Generate multiple initial solutions from different heuristics
    initial_solutions = []
    
    # Heuristic 1: Rank matching (heavy flows to central locations)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy_order = np.argsort(-a_row_sums, kind="stable")
    central_order = np.argsort(b_row_sums, kind="stable")
    p1 = np.empty(n, dtype=np.int64)
    p1[heavy_order] = central_order
    initial_solutions.append(p1)
    
    # Heuristic 2: Inverse rank matching (light flows to central, heavy to periphery)
    light_order = np.argsort(a_row_sums, kind="stable")
    peripheral_order = np.argsort(-b_row_sums, kind="stable")
    p2 = np.empty(n, dtype=np.int64)
    p2[light_order] = peripheral_order
    initial_solutions.append(p2)
    
    # Heuristic 3: Random
    p3 = np.random.permutation(n).astype(np.int64)
    initial_solutions.append(p3)

    # Evaluate and keep the best initial solution
    best_p = None
    best_cost = float('inf')
    
    for p_init in initial_solutions:
        cost = compute_cost(p_init)
        if cost < best_cost:
            best_cost = cost
            best_p = p_init.copy()

    # Run steepest descent local search on the best initial solution
    time_left_for_ls = deadline - 10
    p, current_cost = steepest_descent_local_search(best_p.copy(), best_cost, time_left_for_ls)
    
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()

    # If time remains, try random restarts with perturbation + local search
    while time.time() < deadline - 5:
        p_restart = best_p.copy()
        num_swaps = max(1, n // 10)
        for _ in range(num_swaps):
            i, j = np.random.choice(n, 2, replace=False)
            p_restart[i], p_restart[j] = p_restart[j], p_restart[i]
        
        current_cost_r = compute_cost(p_restart)
        
        time_left = deadline - 5 - time.time()
        if time_left < 1:
            break
        
        # Use first fit for speed on restarts
        p_r, cost_r = first_fit_local_search(p_restart, current_cost_r, time.time() + time_left)
        
        if cost_r < best_cost:
            best_cost = cost_r
            best_p = p_r.copy()

    return best_p.tolist()
