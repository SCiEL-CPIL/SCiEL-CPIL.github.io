"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    """
    Hybrid GRASP + Simulated Annealing for QAP.
    Combines greedy randomized construction, continuous relaxation hints,
    and SA with 2-swap moves for global search.
    """
    n = len(a)
    deadline = time.time() + 115.0  # safety margin before 120s kill

    # Precompute symmetric versions for cost computation
    # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]

    def compute_cost(p):
        """Compute the QAP cost for a given permutation p."""
        B_p = b[np.ix_(p, p)]
        return np.sum(a * B_p)

    def compute_swap_delta(p, i, j):
        """
        Compute the change in cost if we swap facilities i and j in permutation p.
        """
        li = p[i]
        lj = p[j]
        
        # Self terms
        delta = a[i, i] * (b[lj, lj] - b[li, li]) + a[j, j] * (b[li, li] - b[lj, lj])
        
        # Cross terms (i,j) and (j,i)
        delta += a[i, j] * (b[lj, li] - b[li, lj])
        delta += a[j, i] * (b[li, lj] - b[lj, li])
        
        # Terms with k != i, j
        S1_full = np.dot(a[:, i], b[p, lj]) - np.dot(a[:, i], b[p, li])
        S2_full = np.dot(a[i, :], b[lj, p]) - np.dot(a[i, :], b[li, p])
        S3_full = np.dot(a[:, j], b[p, li]) - np.dot(a[:, j], b[p, lj])
        S4_full = np.dot(a[j, :], b[li, p]) - np.dot(a[j, :], b[lj, p])
        
        S1_sub = a[i, i] * (b[li, lj] - b[li, li]) + a[j, i] * (b[lj, li] - b[lj, lj])
        S2_sub = a[i, i] * (b[lj, li] - b[li, li]) + a[i, j] * (b[lj, lj] - b[li, lj])
        S3_sub = a[i, j] * (b[li, li] - b[li, lj]) + a[j, j] * (b[lj, li] - b[lj, lj])
        S4_sub = a[j, i] * (b[li, li] - b[lj, li]) + a[j, j] * (b[li, lj] - b[lj, lj])
        
        delta += (S1_full - S1_sub) + (S2_full - S2_sub) + (S3_full - S3_sub) + (S4_full - S4_sub)
        
        return delta

    def local_search(p, current_cost, deadline):
        """Vectorized 2-swap local search."""
        p = p.copy()
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if time.time() >= deadline:
                    break
                for j in range(i + 1, n):
                    delta = compute_swap_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        improved = True
        return p, current_cost

    def grasp_construct(alpha=0.3):
        """
        Greedy Randomized construction.
        """
        n_facilities = list(range(n))
        n_locations = list(range(n))
        p = np.full(n, -1, dtype=np.int64)
        placed = []
        
        random.shuffle(n_facilities)
        
        for step in range(n):
            if time.time() >= deadline:
                break
            idx = random.randint(0, len(n_facilities) - 1)
            i = n_facilities.pop(idx)
            
            costs = np.zeros(len(n_locations), dtype=np.float64)
            for loc_idx, l in enumerate(n_locations):
                cost = a[i, i] * b[l, l]
                if placed:
                    placed_arr = np.array(placed)
                    p_placed = p[placed_arr]
                    cost += np.dot(a[i, placed_arr], b[l, p_placed])
                    cost += np.dot(a[placed_arr, i], b[p_placed, l])
                costs[loc_idx] = cost
            
            min_cost = np.min(costs)
            max_cost = np.max(costs)
            threshold = min_cost + alpha * (max_cost - min_cost)
            candidates = [loc_idx for loc_idx in range(len(n_locations)) if costs[loc_idx] <= threshold]
            
            chosen_loc_idx = random.choice(candidates)
            l = n_locations.pop(chosen_loc_idx)
            p[i] = l
            placed.append(i)
        
        return p

    def simulated_annealing(p_init, current_cost, deadline, t_start, t_end, cooling):
        """
        Simulated Annealing with 2-swap moves.
        """
        p = p_init.copy()
        cost = current_cost
        t = t_start
        
        while t > t_end and time.time() < deadline:
            # Pick a random pair (i, j)
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
                
            delta = compute_swap_delta(p, i, j)
            
            if delta < 0:
                # Accept
                p[i], p[j] = p[j], p[i]
                cost += delta
            else:
                # Metropolis criterion
                if random.random() < np.exp(-delta / t):
                    p[i], p[j] = p[j], p[i]
                    cost += delta
            
            # Cool down
            t *= cooling
        
        return p, cost

    def continuous_relaxation_hint(p, deadline):
        """
        Use continuous relaxation to generate a hint permutation.
        """
        # Initialize X as a doubly stochastic matrix based on p
        X = np.zeros((n, n), dtype=np.float64)
        X[np.arange(n), p] = 1.0
        
        step_size = 0.01
        
        # Projected Gradient Descent
        for _ in range(20):
            if time.time() > deadline:
                break
            
            # Compute Gradient
            T1 = a @ X
            T2 = T1 @ b.T
            T3 = a.T @ X
            T4 = T3 @ b
            grad = T2 + T4
            
            # Update X
            X = X - step_size * grad
            
            # Project to Birkhoff (simple row/col normalization)
            X = np.maximum(X, 1e-12)
            for _ in range(10):
                r = X.sum(axis=1, keepdims=True)
                r[r == 0] = 1
                X = X / r
                c = X.sum(axis=0, keepdims=True)
                c[c == 0] = 1
                X = X / c
            
            step_size *= 0.9
        
        # Recover integer solution via Hungarian
        cost_matrix = T2 + T4
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        p_hint = col_ind.copy()
        
        return p_hint

    # Main GRASP + SA loop
    best_p = None
    best_cost = np.inf
    
    # First, try the rank-matching heuristic as a starting point
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_init = np.empty(n, dtype=np.int64)
    p_init[heavy] = central
    cost_init = compute_cost(p_init)
    
    if cost_init < best_cost:
        best_cost = cost_init
        best_p = p_init.copy()
    
    # Now do local search on the initial solution
    p_ls, cost_ls = local_search(p_init, cost_init, deadline)
    if cost_ls < best_cost:
        best_cost = cost_ls
        best_p = p_ls.copy()
    
    # GRASP + SA iterations
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Construct a new solution using GRASP
        p_grasp = grasp_construct(alpha=0.2)
        cost_grasp = compute_cost(p_grasp)
        
        # Local search
        p_ls, cost_ls = local_search(p_grasp, cost_grasp, deadline)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
        
        # If we have time, do SA on the best solution
        if time.time() < deadline - 30:
            t_start = max(best_cost * 0.1, 1000.0)
            t_end = max(best_cost * 0.001, 1.0)
            cooling = 0.995
            
            p_sa, cost_sa = simulated_annealing(best_p, best_cost, deadline - 30, t_start, t_end, cooling)
            
            # Local search on SA result
            p_sa_ls, cost_sa_ls = local_search(p_sa, cost_sa, deadline - 5)
            
            if cost_sa_ls < best_cost:
                best_cost = cost_sa_ls
                best_p = p_sa_ls.copy()
        
        # If we're running out of time, break
        if time.time() >= deadline - 5:
            break
    
    # Final polish with local search
    p_final, cost_final = local_search(best_p, best_cost, deadline)
    
    return p_final.tolist()
# EVOLVE-BLOCK-END
