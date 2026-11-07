import numpy as np
import time

def simulate_aco_for_heuristic(distance_matrix, heuristic, n_ants=20, n_iter=30, seed=42):
    n = distance_matrix.shape[0]
    np.random.seed(seed)
    
    h = heuristic.copy()
    h[~np.isfinite(h)] = 1e-9
    h[h <= 0] = 1e-9
    
    tau = np.ones((n, n))
    global_best_length = np.inf
    
    for it in range(n_iter):
        deposits = np.zeros((n, n))
        
        for ant in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            current = 0
            tour_length = 0.0
            tour = [0]
            
            for _ in range(n - 1):
                # Vectorized probability calculation
                probs = tau[current, :] * h[current, :]
                probs[visited] = 0
                
                total = np.sum(probs)
                if total == 0:
                    unvisited_idx = np.where(~visited)[0]
                    next_node = np.random.choice(unvisited_idx)
                else:
                    probs = probs / total
                    next_node = np.random.choice(n, p=probs)
                
                tour_length += distance_matrix[current, next_node]
                tour.append(next_node)
                visited[next_node] = True
                current = next_node
                
            tour_length += distance_matrix[current, 0]
            tour.append(0)
            
            if tour_length < global_best_length:
                global_best_length = tour_length
            
            deposit_val = 1.0 / tour_length
            for k in range(len(tour) - 1):
                i = tour[k]
                j = tour[k+1]
                deposits[i, j] += deposit_val
        
        tau = tau * 0.9 + deposits
    
    return global_best_length

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = 320.0  # Safety margin before 330s kill
    
    # Ensure float type and handle edge cases
    d = np.asarray(distance_matrix, dtype=float)
    if not np.all(np.isfinite(d)):
        # If input is invalid, return a safe default
        d = np.nan_to_num(d, nan=1.0, posinf=1.0, neginf=1.0)
    
    # Compute inverse distance
    # Add epsilon to avoid division by zero on diagonal
    inv_dist = 1.0 / (d + 1e-9)
    
    # Candidate alphas for eta = (1/d)^alpha
    # The current program uses alpha=2 (1/d^2). 
    # Inspirations suggest searching around this. 
    # Let's search a range of alphas to find the optimal exponent.
    # Based on the current score (1.25) vs baseline (1.0), the current alpha=2 is better.
    # Let's refine around alpha=2, but also check a broader range to be safe.
    
    # Use a logarithmic scale for alpha to cover both small and large exponents efficiently
    # or a linear scale if we expect the optimum to be in a specific range.
    # Given alpha=2 works well, let's search [1, 5] with fine granularity.
    alphas = np.linspace(1.0, 5.0, 41)
    
    best_alpha = 2.0
    best_avg_score = np.inf
    best_heuristic = inv_dist ** 2.0  # Default to current best
    
    # Use a few seeds to estimate performance robustly
    seeds = [0, 1, 2, 3, 4]
    
    # Pre-compute heuristic candidates to avoid recomputation inside the loop if possible?
    # No, each alpha gives a different matrix. But we can compute them quickly.
    
    # Sort alphas by proximity to current best (2.0) to find good solutions early?
    # Or just iterate. Given 41 alphas * 5 seeds = 205 simulations.
    # Each simulation is fast (50 nodes, 30 iters, 20 ants).
    # Let's estimate time: 50*30*20*50 operations ~ 1.5M ops per sim? No, inner loop is O(N^2) per step?
    # Actually, the inner loop is O(N) for probability calculation (vectorized).
    # So one sim is roughly 30 * 20 * 50 = 30,000 steps. Very fast.
    # 205 sims should be well within 320s.
    
    for alpha in alphas:
        if time.time() > deadline:
            break
            
        h_candidate = np.power(inv_dist, alpha)
        
        total_score = 0.0
        for seed in seeds:
            score = simulate_aco_for_heuristic(d, h_candidate, seed=seed)
            total_score += score
            
        avg_score = total_score / len(seeds)
        
        if avg_score < best_avg_score:
            best_avg_score = avg_score
            best_alpha = alpha
            best_heuristic = h_candidate
            
    # If we used all time, return best found.
    # If we finished early, we could potentially refine further, but the linear scan is likely sufficient.
    # Let's do a local refinement if we have time left.
    remaining_time = deadline - time.time()
    if remaining_time > 20.0:
        # Refine around best_alpha
        local_alphas = np.linspace(best_alpha - 0.5, best_alpha + 0.5, 21)
        for alpha in local_alphas:
            if time.time() > deadline:
                break
            h_candidate = np.power(inv_dist, alpha)
            total_score = 0.0
            for seed in seeds:
                score = simulate_aco_for_heuristic(d, h_candidate, seed=seed)
                total_score += score
            avg_score = total_score / len(seeds)
            if avg_score < best_avg_score:
                best_avg_score = avg_score
                best_alpha = alpha
                best_heuristic = h_candidate

    return best_heuristic
