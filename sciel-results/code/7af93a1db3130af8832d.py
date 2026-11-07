import numpy as np
import time

def _simulate_aco(prizes, distance_matrix, max_len, heuristic, num_ants=20, num_iters=30, seed=42):
    """
    Vectorized simulation of the ACO process.
    Returns the best prize collected by any ant over all iterations.
    """
    n = len(prizes)
    H = np.where(heuristic > 0, heuristic, 1e-9)
    
    best_prize = 0.0
    rng = np.random.default_rng(seed)
    
    d_to_start = distance_matrix[:, 0]
    
    for it in range(num_iters):
        best_iter_prize = 0.0
        
        for _ in range(num_ants):
            current_node = 0
            visited_mask = np.zeros(n, dtype=bool)
            visited_mask[0] = True
            path_len = 0.0
            total_prize = prizes[0]
            
            while True:
                remaining_budget = max_len - path_len
                d_from_current = distance_matrix[current_node, :]
                
                # Vectorized eligibility check
                # Condition: not visited AND (dist[i, j] + dist[j, 0] <= remaining_budget)
                valid_indices = np.where(
                    (~visited_mask) & 
                    (d_from_current + d_to_start <= remaining_budget)
                )[0]
                
                if len(valid_indices) == 0:
                    break
                
                h_vals = H[current_node, valid_indices]
                total_h = np.sum(h_vals)
                
                if total_h <= 0:
                    break
                    
                probs = h_vals / total_h
                
                # Vectorized choice
                next_node = valid_indices[rng.choice(len(valid_indices), p=probs)]
                
                path_len += distance_matrix[current_node, next_node]
                total_prize += prizes[next_node]
                visited_mask[next_node] = True
                current_node = next_node
            
        best_iter_prize = max(best_iter_prize, total_prize)
        best_prize = max(best_prize, best_iter_prize)

    return best_prize

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    deadline = start_time + 290  # 40s safety margin
    
    n = len(prizes)
    
    # 1. Initialize with a robust multi-criteria heuristic
    # Blend prize attractiveness and distance penalty
    # H[i, j] = (prize[j] + eps) / (dist[i, j] + eps)^alpha
    # Let's try a few initial candidates and pick the best
    candidates = []
    
    # Standard
    H1 = np.where(prizes[None, :] > 0, prizes[None, :], 1e-9) / (distance_matrix + 0.1)
    np.fill_diagonal(H1, 1e-9)
    candidates.append(H1)
    
    # Prize squared
    H2 = np.where(prizes[None, :] > 0, prizes[None, :]**2, 1e-9) / (distance_matrix + 0.1)
    np.fill_diagonal(H2, 1e-9)
    candidates.append(H2)
    
    # Distance penalized more
    H3 = np.where(prizes[None, :] > 0, prizes[None, :], 1e-9) / (distance_matrix + 0.1)**2
    np.fill_diagonal(H3, 1e-9)
    candidates.append(H3)
    
    best_h = candidates[0]
    best_score = _simulate_aco(prizes, distance_matrix, max_len, best_h)
    
    for h in candidates[1:]:
        if time.time() > deadline - 5:
            break
        s = _simulate_aco(prizes, distance_matrix, max_len, h)
        if s > best_score:
            best_score = s
            best_h = h
    
    # 2. Direct Stochastic Coordinate Ascent (Hill Climbing)
    # We operate on log-space to ensure positivity and allow multiplicative updates
    # H = exp(log_H)
    
    # Ensure best_h is strictly positive for log
    log_h = np.log(np.where(best_h > 0, best_h, 1e-9))
    
    current_score = best_score
    current_log_h = log_h.copy()
    
    # Adaptive step size
    step_size = 0.5  # Start relatively large to explore
    no_improve_count = 0
    max_no_improve = 20
    
    # Pre-compute valid edges? 
    # An edge (i, j) is only useful if d[i, j] + d[j, 0] <= max_len AND d[i, 0] + d[i, j] <= max_len? 
    # Actually, the constraint is path_len_so_far + d[i, j] + d[j, 0] <= max_len.
    # Since path_len_so_far >= 0, a necessary condition for (i, j) to ever be traversed is d[i, j] + d[j, 0] <= max_len.
    # Also, to reach i, we need d[0, i] <= max_len (roughly, since path is 0->...->i).
    # Let's just restrict perturbations to edges where d[i, j] + d[j, 0] <= max_len.
    # This saves time and avoids useless perturbations.
    
    valid_edges_mask = (distance_matrix + distance_matrix[:, 0:1]) <= max_len
    valid_edges_mask = np.where(valid_edges_mask, 1, 0)
    # Don't perturb diagonal or start node outgoing edges? 
    # Actually, edges from 0 are crucial.
    # Let's just use the mask.
    
    valid_edge_indices = np.argwhere(valid_edges_mask > 0)
    num_valid_edges = len(valid_edge_indices)
    
    if num_valid_edges == 0:
        return best_h
    
    # Main loop
    while time.time() < deadline - 2:
        # If no improvement for a while, reduce step size
        if no_improve_count > max_no_improve:
            step_size *= 0.8
            no_improve_count = 0
            if step_size < 0.01:
                break
        
        # Pick a random subset of valid edges to perturb
        # Batch size: try 5-10 edges at once to save overhead? 
        # Or single edge for finer control? 
        # Single edge is safer for local optima, but slower. 
        # Let's try perturbing a small random batch (e.g., 5 edges) to explore neighborhood.
        
        batch_size = min(5, num_valid_edges)
        # Randomly select batch_size edges
        perm = np.random.permutation(num_valid_edges)
        selected_edges = valid_edge_indices[perm[:batch_size]]
        
        # Generate random perturbations in log-space
        # Gaussian noise
        noise = np.random.normal(0, step_size, size=batch_size)
        
        new_log_h = current_log_h.copy()
        for k, (i, j) in enumerate(selected_edges):
            new_log_h[i, j] += noise[k]
        
        # Convert back to heuristic
        new_h = np.exp(new_log_h)
        new_h = np.where(new_h > 0, new_h, 1e-9)
        # Ensure diagonal is small (though it shouldn't be selected if mask excludes it, but safe)
        np.fill_diagonal(new_h, 1e-9)
        
        new_score = _simulate_aco(prizes, distance_matrix, max_len, new_h)
        
        if new_score > current_score:
            current_score = new_score
            current_log_h = new_log_h
            best_h = new_h
            no_improve_count = 0
            # Increase step size slightly if improving? (Simulated annealing style)
            # Or keep it constant? Let's keep it constant to avoid oscillation.
        else:
            no_improve_count += 1
            
    return best_h
