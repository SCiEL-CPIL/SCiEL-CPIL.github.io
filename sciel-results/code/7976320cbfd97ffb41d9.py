"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _generate_tour(prizes, distance_matrix, max_len, noise_scale=0.0, seed=0):
    """
    Generates a tour using a greedy heuristic with optional noise.
    Returns a list of node indices representing the tour.
    """
    n = len(prizes)
    current = 0
    length = 0.0
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    tour = [0]
    
    # Use a local RNG for reproducibility within the call if seed is provided
    rng = np.random.default_rng(seed)
    
    while True:
        best_j = -1
        best_score = -np.inf
        
        # Vectorized search for next node
        # Calculate remaining budget for the leg: max_len - length - d[j, 0]
        # We need: length + d[current, j] + d[j, 0] <= max_len
        # => d[current, j] + d[j, 0] <= max_len - length
        
        remaining_budget = max_len - length
        if remaining_budget <= 0:
            break
            
        # Candidates must not be visited
        candidates = np.where(~visited)[0]
        if len(candidates) == 0:
            break
            
        # Filter candidates by feasibility
        # d[current, j] + d[j, 0] <= remaining_budget
        dist_from_current = distance_matrix[current, candidates]
        dist_to_start = distance_matrix[candidates, 0]
        total_leg_cost = dist_from_current + dist_to_start
        
        feasible_mask = total_leg_cost <= remaining_budget
        feasible_candidates = candidates[feasible_mask]
        
        if len(feasible_candidates) == 0:
            break
            
        # Calculate scores for feasible candidates
        # Base score: Prize / Distance
        # Add noise to diversify
        prizes_c = prizes[feasible_candidates]
        dists_c = dist_from_current[feasible_mask]
        
        # Avoid division by zero
        dists_safe = np.where(dists_c > 1e-9, dists_c, 1e-9)
        
        scores = prizes_c / dists_safe
        
        # Add noise if noise_scale > 0
        if noise_scale > 0:
            noise = rng.normal(0, noise_scale, size=len(feasible_candidates))
            scores += noise
            
        # Find best
        if len(scores) > 0:
            max_idx = np.argmax(scores)
            if scores[max_idx] > best_score:
                best_score = scores[max_idx]
                best_j = feasible_candidates[max_idx]
                
        if best_j == -1:
            break
            
        # Update state
        tour.append(best_j)
        length += distance_matrix[current, best_j]
        visited[best_j] = True
        current = best_j
        
    return tour

def heuristics(prizes, distance_matrix, max_len):
    n = len(prizes)
    
    # 1. Generate a set of diverse tours using the greedy heuristic
    num_tours = 20
    # Use different noise scales and seeds to get variety
    noise_scales = [0.0, 0.1, 0.2, 0.5, 1.0]
    tours = []
    
    for i in range(num_tours):
        noise = noise_scales[i % len(noise_scales)]
        seed = i * 12345 + 6789
        tour = _generate_tour(prizes, distance_matrix, max_len, noise_scale=noise, seed=seed)
        tours.append(tour)
        
    # 2. Build a frequency matrix based on these tours
    # H_freq[i, j] counts how many times edge (i, j) appears in the tours
    H_freq = np.zeros((n, n), dtype=float)
    
    for tour in tours:
        for k in range(len(tour) - 1):
            u = tour[k]
            v = tour[k+1]
            H_freq[u, v] += 1.0
            # If distance matrix is symmetric, we might want to consider reverse too?
            # ACO is directed. The tour is a path 0->...->0.
            # The edge (u,v) is used. The edge (v,u) is not used in this direction.
            # However, if the problem is symmetric, visiting v then u is also valid.
            # But the heuristic should guide the *current* direction.
            # Let's stick to directed edges found in the tours.
            
    # 3. Create a base distance heuristic
    # H_dist[i, j] = 1 / (d[i, j] + epsilon)
    # This ensures that even if no tour uses an edge, it has some attractiveness if close.
    dist_safe = distance_matrix.copy()
    np.fill_diagonal(dist_safe, np.inf)
    H_dist = 1.0 / dist_safe
    # Handle inf -> 0
    H_dist[~np.isfinite(H_dist)] = 0.0
    
    # 4. Combine
    # We want to strongly favor edges that appeared in good tours.
    # But we also want to maintain some connectivity.
    # Normalize H_freq to be in a similar range as H_dist?
    # H_freq values are small integers (0-20). H_dist values are ~1.
    # Let's scale H_freq.
    max_freq = np.max(H_freq)
    if max_freq > 0:
        H_freq_norm = H_freq / max_freq
    else:
        H_freq_norm = np.zeros_like(H_freq)
        
    # Blend: 80% tour frequency, 20% distance
    H = 0.8 * H_freq_norm + 0.2 * H_dist
    
    # 5. Apply feasibility mask
    # If node j is not feasible (d[0,j] + d[j,0] > max_len), it can't be in any tour.
    start_end_cost = distance_matrix[0, :] + distance_matrix[:, 0]
    feasible_nodes = start_end_cost <= max_len
    
    # If node j is not feasible, H[i, j] should be very low (but not 0 to avoid issues? 
    # Task says non-positive becomes 1e-9. So 0 is fine, it becomes 1e-9).
    H[:, ~feasible_nodes] = 0.0
    
    # Diagonal should be 0 (can't visit self)
    np.fill_diagonal(H, 0.0)
    
    # 6. Handle non-positive and non-finite values
    # Task: "Heuristic entries that are not positive become 1e-9"
    H = np.where(H > 0, H, 1e-9)
    
    # Final check for finiteness
    if not np.all(np.isfinite(H)):
        H = np.ones((n, n)) * 1e-9
        np.fill_diagonal(H, 1e-9)
        
    return H
# EVOLVE-BLOCK-END
