"""
Strategy: Full Pipeline Lookahead
The previous approach simulated only the first step (2-opt on perturbed matrix) 
and evaluated the intermediate tour's true length. However, the evaluator performs 
a second 2-opt on the true distances. A tour that looks good after the first step 
might be in a "dead end" for the second step, while a tour that looks worse 
intermediately might lead to a much better final result after the second 2-opt.

This program simulates the full evaluator loop for a set of candidate perturbation 
matrices. It returns the matrix that leads to the best final tour length when 
measured against the true distances after both 2-opt passes.
"""
import numpy as np
import time

def _fast_2opt(matrix, tour, max_passes=10):
    """
    Fast 2-opt implementation. Returns the best tour and its length
    measured using the given matrix.
    """
    n = len(tour)
    current_tour = list(tour)
    
    def tour_length(t):
        return sum(matrix[t[i], t[(i + 1) % n]] for i in range(n))
    
    current_len = tour_length(current_tour)
    
    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                if j == i + 1:
                    continue
                if i == 0 and j == n - 1:
                    continue
                
                a = current_tour[i - 1]
                b = current_tour[i]
                c = current_tour[j]
                d = current_tour[(j + 1) % n]
                
                old_cost = matrix[a, b] + matrix[c, d]
                new_cost = matrix[a, c] + matrix[b, d]
                
                if new_cost < old_cost - 1e-10:
                    current_tour[i:j+1] = current_tour[i:j+1][::-1]
                    current_len = new_cost - old_cost + current_len
                    improved = True
                    
        if not improved:
            break
            
    return current_tour, current_len

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Generate multiple candidate perturbation matrices, simulate the full evaluator
    pipeline (2-opt on candidate, then 2-opt on true) for each, and return the
    matrix that leads to the best final tour length on true distances.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Leave 40s margin for 330s total limit
    
    n = len(local_opt_tour)
    
    # Extract edges from the current tour
    tour_edges = []
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        if u > v:
            u, v = v, u
        tour_edges.append((u, v))
    
    # We will try a set of (alpha, beta) penalty configurations.
    # The penalty is applied to the edges in the current tour.
    # P(i, j) = alpha * (1 + beta * edge_n_used[i, j])
    
    # Define a grid of parameters to explore.
    # We want a mix of "strong" kicks (high alpha) and "adaptive" kicks (high beta).
    alphas = [0.5, 1.0, 2.0, 5.0, 10.0]
    betas = [0.0, 0.5, 1.0, 2.0]
    
    best_true_len = float('inf')
    best_matrix = edge_distance.copy()
    
    # Helper to evaluate a candidate matrix through the full pipeline
    def evaluate_candidate(cand_matrix):
        # 1. Run 2-opt on cand_matrix (simulating the evaluator's first step)
        # The evaluator uses max_passes=10. We should match this for accurate simulation.
        tour1, _ = _fast_2opt(cand_matrix, local_opt_tour, max_passes=10)
        
        # 2. Run 2-opt on edge_distance (true) starting from tour1 (simulating the evaluator's second step)
        tour2, _ = _fast_2opt(edge_distance, tour1, max_passes=10)
        
        # 3. Calculate the true length of the final tour
        true_len = 0.0
        for i in range(n):
            true_len += edge_distance[tour2[i], tour2[(i + 1) % n]]
        
        return true_len

    # Iterate through the parameter grid
    for alpha in alphas:
        if time.time() > deadline:
            break
            
        for beta in betas:
            if time.time() > deadline:
                break
                
            cand_matrix = edge_distance.copy()
            
            # Apply penalty to edges in the current tour
            for u, v in tour_edges:
                count = edge_n_used[u, v]
                penalty = alpha * (1.0 + beta * count)
                cand_matrix[u, v] += penalty
                cand_matrix[v, u] += penalty
            
            true_len = evaluate_candidate(cand_matrix)
            
            if true_len < best_true_len:
                best_true_len = true_len
                best_matrix = cand_matrix
                
    # If we have time left, try some randomized perturbations
    # This helps escape from cases where the grid search might miss a "lucky" kick
    if time.time() < deadline - 5:
        for _ in range(5):
            if time.time() > deadline:
                break
            alpha = np.random.uniform(1.0, 10.0)
            beta = np.random.uniform(0.0, 2.0)
            
            cand_matrix = edge_distance.copy()
            for u, v in tour_edges:
                count = edge_n_used[u, v]
                penalty = alpha * (1.0 + beta * count)
                cand_matrix[u, v] += penalty
                cand_matrix[v, u] += penalty
                
            true_len = evaluate_candidate(cand_matrix)
            
            if true_len < best_true_len:
                best_true_len = true_len
                best_matrix = cand_matrix

    return best_matrix
