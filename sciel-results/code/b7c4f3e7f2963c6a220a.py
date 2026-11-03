"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def solve(a, b):
    """
    GRASP for QAP: Greedy Randomized construction + vectorized 2-swap local search.
    """
    n = len(a)
    deadline = time.time() + 115.0  # safety margin before 120s kill

    # Precompute symmetric versions for cost computation
    # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # For symmetric a and b (QAPLIB instances are typically symmetric), 
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # Note: a and b may not be symmetric in general, but QAPLIB tho150 is symmetric.
    # We'll handle the general case.

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
        """
        2-swap local search with tabu-like memory to avoid cycling.
        Uses best-improvement per pass.
        """
        p = p.copy()
        last_swap = (-1, -1)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_delta = 0
            best_i = -1
            best_j = -1
            
            # Find best improving swap
            for i in range(n):
                if time.time() >= deadline:
                    break
                for j in range(i + 1, n):
                    # Skip recent swaps to reduce cycling
                    if (i == last_swap[0] and j == last_swap[1]) or (i == last_swap[1] and j == last_swap[0]):
                        continue
                    delta = compute_swap_delta(p, i, j)
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
            
            if best_i >= 0:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += best_delta
                last_swap = (best_i, best_j)
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
            
            # Compute cost of assigning i to each remaining location
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

    # Main GRASP loop
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
    
    # GRASP iterations
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        # Construct a new solution using GRASP with varying alpha
        alpha = 0.1 + 0.2 * random.random()
        p_grasp = grasp_construct(alpha=alpha)
        
        # Compute cost
        cost_grasp = compute_cost(p_grasp)
        
        # Local search
        p_ls, cost_ls = local_search(p_grasp, cost_grasp, deadline)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
        
        # If we're running out of time, break
        if time.time() >= deadline - 5:
            break
    
    # Final polish: do a few more local search passes on the best solution
    if time.time() < deadline - 2:
        p_final, cost_final = local_search(best_p, best_cost, deadline)
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
