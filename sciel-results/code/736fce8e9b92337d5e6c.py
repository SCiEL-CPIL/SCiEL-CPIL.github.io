import numpy as np
import time

def _fast_aco_simulate(distance_matrix, heuristic, n_ants=15, n_iterations=15, rho=0.9, seed=42):
    """
    Fast ACO simulation to evaluate heuristic quality.
    Returns the minimum tour length found.
    """
    n = distance_matrix.shape[0]
    tau = np.ones((n, n))
    rng = np.random.default_rng(seed)
    
    best_tour_len = float('inf')
    
    for it in range(n_iterations):
        tour_lengths = []
        for ant in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            current = 0
            tour_len = 0.0
            tour_path = [0]
            
            for _ in range(n - 1):
                # Vectorized probability calculation
                probs = tau[current, :] * heuristic[current, :]
                probs[visited] = 0.0
                
                total_prob = probs.sum()
                if total_prob < 1e-10:
                    unvisited_indices = np.where(~visited)[0]
                    next_node = unvisited_indices[rng.integers(len(unvisited_indices))]
                else:
                    probs = probs / total_prob
                    next_node = rng.choice(n, p=probs)
                
                tour_len += distance_matrix[current, next_node]
                visited[next_node] = True
                tour_path.append(next_node)
                current = next_node
            
            tour_len += distance_matrix[current, 0]
            tour_lengths.append(tour_len)
            
            deposit = 1.0 / tour_len
            # Vectorized pheromone update (approximate for speed, or loop if n is small enough)
            # Since n=50, loop is fine and safer for directionality
            for i in range(n):
                u = tour_path[i]
                v = tour_path[(i + 1) % n]
                tau[u, v] += deposit
        
        tau *= rho
        
        current_best = min(tour_lengths)
        if current_best < best_tour_len:
            best_tour_len = current_best
            
    return best_tour_len

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 280.0
    
    n = distance_matrix.shape[0]
    dm = distance_matrix.copy()
    
    # 1. Precompute geometric features
    # Inverse distance (base)
    safe_dist = dm.copy()
    np.fill_diagonal(safe_dist, 1.0)
    safe_dist[safe_dist < 1e-9] = 1e-9
    inv_dist = 1.0 / safe_dist
    
    # Nearest neighbors (k=5)
    # Calculate distances from each node to all others
    # dm has 0 on diagonal, so we can just find smallest values
    # But we need to exclude self.
    # For each row i, find the 5 smallest distances
    nn_mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        row = dm[i].copy()
        row[i] = np.inf
        # Get indices of 5 smallest
        nn_indices = np.argsort(row)[:5]
        nn_mask[i, nn_indices] = True
    
    nn_bonus = np.where(nn_mask, 1.0, 0.0)
    
    # Angular penalty (turning cost)
    # For each triplet (i, j, k), calculate the angle or cross product magnitude
    # We want to penalize sharp turns (small angles) to encourage smoother paths
    # Vectorized calculation of angles is tricky for all pairs.
    # Approximation: Use the distance between i and k.
    # If i and k are close, the path i->j->k is a "sharp turn" or backtrack.
    # If i and k are far, it's a straight line.
    # Actually, for TSP, we want to avoid crossing edges.
    # A simpler heuristic feature: "straightness"
    # Let's use the distance d(i,k) as a proxy.
    # If d(i,k) is small, the turn is sharp.
    # We want to *decrease* attractiveness for sharp turns.
    # So, penalty = 1 / (d(i,k) + eps) ? No, that increases for sharp turns.
    # We want heuristic to be LOW for sharp turns.
    # So, feature = d(i,k). Larger d(i,k) is better (straighter).
    # But this is a triplet feature, not a pair feature.
    # The heuristic matrix H[i,j] is applied when at i, choosing j.
    # The "previous" node is not known in the matrix H itself.
    # So we cannot directly encode "turning cost" into a static H[i,j] matrix
    # because H[i,j] doesn't know where we came from.
    # 
    # WAIT. The contract says: "picks the next unvisited node with probability proportional to pheromone[i, j] * heuristic[i, j]".
    # The heuristic is a matrix H[i,j]. It does NOT depend on the previous node.
    # Therefore, angular/turning heuristics that depend on the path history CANNOT be directly encoded in H[i,j].
    # 
    # So, my "angular penalty" idea is invalid for this specific interface.
    # I must stick to pairwise features.
    
    # Revised Features:
    # 1. Inverse Distance (inv_dist)
    # 2. Nearest Neighbor Bonus (nn_bonus) - encourages local moves
    # 3. "Global" connectivity? Maybe inverse of max distance?
    # 4. Power of inverse distance: inv_dist ** p
    
    # Let's define a parameterized family:
    # H[i,j] = (inv_dist[i,j] ** p) * (1 + alpha * nn_bonus[i,j])
    # Where p > 0 and alpha >= 0.
    
    # We will search over p and alpha.
    
    best_score = float('inf')
    best_p = 1.0
    best_alpha = 0.0
    best_heuristic = inv_dist.copy()
    
    # Initial candidates
    p_values = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    alpha_values = [0.0, 0.5, 1.0, 2.0]
    
    # Grid search
    for p in p_values:
        if time.time() > deadline - 30.0:
            break
        for alpha in alpha_values:
            if time.time() > deadline - 30.0:
                break
            
            # Construct heuristic
            h = (inv_dist ** p) * (1.0 + alpha * nn_bonus)
            h[h <= 0] = 1e-9
            h[~np.isfinite(h)] = 1e-9
            
            # Evaluate
            score = _fast_aco_simulate(distance_matrix, h, n_ants=15, n_iterations=15)
            
            if score < best_score:
                best_score = score
                best_p = p
                best_alpha = alpha
                best_heuristic = h.copy()
                
    # Local refinement around best found
    if time.time() < deadline - 20.0:
        # Refine p
        p_step = 0.5
        for p in np.linspace(best_p - p_step, best_p + p_step, 5):
            if p <= 0: continue
            if time.time() > deadline - 20.0:
                break
            for alpha in np.linspace(max(0, best_alpha - 0.5), best_alpha + 0.5, 5):
                if time.time() > deadline - 20.0:
                    break
                
                h = (inv_dist ** p) * (1.0 + alpha * nn_bonus)
                h[h <= 0] = 1e-9
                h[~np.isfinite(h)] = 1e-9
                
                score = _fast_aco_simulate(distance_matrix, h, n_ants=15, n_iterations=15)
                
                if score < best_score:
                    best_score = score
                    best_p = p
                    best_alpha = alpha
                    best_heuristic = h.copy()
                    
    # Final sanity checks
    best_heuristic[best_heuristic <= 0] = 1e-9
    best_heuristic[~np.isfinite(best_heuristic)] = 1e-9
    
    return best_heuristic
