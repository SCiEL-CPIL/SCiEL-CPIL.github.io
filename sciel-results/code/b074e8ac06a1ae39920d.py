import numpy as np
import time

def _simulate_aco(prizes, distance_matrix, max_len, heuristic, num_ants=20, num_iters=30, seed=42):
    """
    Simulates the ACO process to evaluate a heuristic.
    This function is kept identical to the task contract's implied simulator.
    """
    n = len(prizes)
    H = np.where(heuristic > 0, heuristic, 1e-9)
    
    best_prize = 0
    rng = np.random.default_rng(seed)
    
    d_to_start = distance_matrix[:, 0]
    
    for it in range(num_iters):
        best_iter_prize = 0
        
        for _ in range(num_ants):
            current_node = 0
            visited_mask = np.zeros(n, dtype=bool)
            visited_mask[0] = True
            path_len = 0.0
            total_prize = prizes[0]
            
            while True:
                remaining_budget = max_len - path_len
                d_from_current = distance_matrix[current_node, :]
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
    deadline = start_time + 290
    
    n = len(prizes)
    eps = 1e-9
    
    # 1. Generate a set of candidate paths using a diverse set of greedy heuristics.
    # We don't just use one "best" greedy path, but a set of paths that are
    # "greedy" according to different criteria (e.g., max prize, min distance, 
    # max prize/distance, etc.).
    
    candidate_paths = []
    
    # Helper to generate a greedy path given a scoring function for edges
    def generate_greedy_path(score_func):
        current_node = 0
        visited_mask = np.zeros(n, dtype=bool)
        visited_mask[0] = True
        path_len = 0.0
        path = [0]
        
        while True:
            remaining_budget = max_len - path_len
            d_from_current = distance_matrix[current_node, :]
            valid_indices = np.where(
                (~visited_mask) & 
                (d_from_current + distance_matrix[:, 0] <= remaining_budget)
            )[0]
            
            if len(valid_indices) == 0:
                break
            
            # Score valid edges
            scores = np.array([score_func(current_node, j) for j in valid_indices])
            
            # Handle case where all scores are -inf or 0
            if np.all(scores <= 0):
                break
                
            # Pick the best
            best_idx = valid_indices[np.argmax(scores)]
            
            path.append(best_idx)
            path_len += distance_matrix[current_node, best_idx]
            visited_mask[best_idx] = True
            current_node = best_idx
            
        return path

    # Define different scoring functions
    # 1. Pure Prize
    def score_prize(i, j):
        return prizes[j]
        
    # 2. Prize / Distance
    def score_prize_dist(i, j):
        d = distance_matrix[i, j]
        if d < 1e-6: return 0
        return prizes[j] / d
        
    # 3. Prize / (Distance + small)
    def score_prize_dist_smooth(i, j):
        d = distance_matrix[i, j]
        return prizes[j] / (d + 0.1)
        
    # 4. (Prize + 0.5) / (Distance + 0.1)
    def score_prize_offset(i, j):
        d = distance_matrix[i, j]
        return (prizes[j] + 0.5) / (d + 0.1)
        
    # 5. Prize - Distance
    def score_prize_minus_dist(i, j):
        return prizes[j] - distance_matrix[i, j]
        
    # 6. Prize / (Distance ** 0.5)
    def score_prize_sqrt(i, j):
        d = distance_matrix[i, j]
        if d < 1e-6: return 0
        return prizes[j] / np.sqrt(d)
        
    # 7. Prize * (1 / (Distance + 0.2))
    def score_prize_inv_dist(i, j):
        return prizes[j] / (distance_matrix[i, j] + 0.2)

    scorers = [
        score_prize,
        score_prize_dist,
        score_prize_dist_smooth,
        score_prize_offset,
        score_prize_minus_dist,
        score_prize_sqrt,
        score_prize_inv_dist
    ]
    
    for s in scorers:
        p = generate_greedy_path(s)
        if len(p) > 1:
            candidate_paths.append(p)
            
    # Also add some random paths to ensure diversity? 
    # Maybe not necessary if we have enough greedy variations.
    # Let's add a few random walks to be safe, constrained by budget.
    rng_gen = np.random.default_rng(123)
    for _ in range(5):
        current_node = 0
        visited_mask = np.zeros(n, dtype=bool)
        visited_mask[0] = True
        path_len = 0.0
        path = [0]
        
        while True:
            remaining_budget = max_len - path_len
            d_from_current = distance_matrix[current_node, :]
            valid_indices = np.where(
                (~visited_mask) & 
                (d_from_current + distance_matrix[:, 0] <= remaining_budget)
            )[0]
            
            if len(valid_indices) == 0:
                break
            
            # Randomly pick from valid
            next_node = valid_indices[rng_gen.integers(0, len(valid_indices))]
            
            path.append(next_node)
            path_len += distance_matrix[current_node, next_node]
            visited_mask[next_node] = True
            current_node = next_node
            
        if len(path) > 1:
            candidate_paths.append(path)

    # 2. Build the Heuristic Matrix based on these paths.
    # Strategy: The heuristic H[i, j] should be high if the edge (i, j) is 
    # frequently used in high-quality paths.
    # We will weight each path by its total prize.
    
    H = np.full((n, n), 1e-9)
    
    # Calculate total prize for each path
    path_prizes = []
    for p in candidate_paths:
        total_p = sum(prizes[node] for node in p)
        path_prizes.append(total_p)
        
    # Normalize path prizes to weights? Or just sum them?
    # Summing them means edges in high-prize paths get higher weight.
    # This effectively encodes "which edges are good" directly.
    
    for p, weight in zip(candidate_paths, path_prizes):
        if len(p) < 2:
            continue
        for k in range(len(p) - 1):
            i, j = p[k], p[k+1]
            H[i, j] += weight
            
    # 3. Refinement: 
    # The simple sum might be too coarse. 
    # Let's try a "Lookahead" refinement.
    # For each node i, and each valid next node j, we can estimate the 
    # "quality" of taking j by looking at the best continuation from j.
    # However, that's expensive. 
    # 
    # Alternative: Use the ACO simulator itself to evaluate the "Path-Encoded" heuristic 
    # and then do a few steps of "Local Search" on the heuristic matrix itself?
    # No, that's too complex.
    #
    # Let's try a hybrid: 
    # The base heuristic is the path-encoded one.
    # But we can also blend it with a standard prize/distance heuristic 
    # to handle cases where the greedy paths missed a good edge.
    
    # Standard Heuristic: H_std[i, j] = prize[j] / (dist[i, j] + eps)
    H_std = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j: continue
            d = distance_matrix[i, j]
            if d < 1e-6: continue
            H_std[i, j] = prizes[j] / (d + 0.01)
            
    # Normalize H_std and H_path to have similar scales
    # Max normalize
    max_h_path = np.max(H)
    if max_h_path > 0:
        H_path_norm = H / max_h_path
    else:
        H_path_norm = H
        
    max_h_std = np.max(H_std)
    if max_h_std > 0:
        H_std_norm = H_std / max_h_std
    else:
        H_std_norm = H_std

    # Try a few blends
    best_score = -1
    best_h = None
    
    blends = [0.0, 0.2, 0.5, 0.8, 1.0]
    
    for alpha in blends:
        if time.time() > deadline:
            break
            
        H_candidate = (1 - alpha) * H_path_norm + alpha * H_std_norm
        
        # Ensure positive
        H_candidate = np.where(H_candidate > 0, H_candidate, 1e-9)
        
        score = _simulate_aco(prizes, distance_matrix, max_len, H_candidate)
        
        if score > best_score:
            best_score = score
            best_h = H_candidate
            
    # If we have time, try to optimize the blend parameter or even 
    # do a random perturbation of the path-encoded heuristic.
    
    if time.time() < deadline - 20 and best_h is not None:
        # Try to refine the best blend found
        # Or try a few random perturbations of the path weights?
        # Let's try a simple coordinate descent on the blend alpha
        # Find the best alpha from the grid, then refine around it.
        
        # Re-run to find which alpha was best
        best_alpha = blends[0]
        best_score_alpha = -1
        for alpha in blends:
            H_candidate = (1 - alpha) * H_path_norm + alpha * H_std_norm
            H_candidate = np.where(H_candidate > 0, H_candidate, 1e-9)
            score = _simulate_aco(prizes, distance_matrix, max_len, H_candidate)
            if score > best_score_alpha:
                best_score_alpha = score
                best_alpha = alpha
                
        # Refine alpha
        current_alpha = best_alpha
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            # Try alpha +/- 0.1
            for delta_a in [0.1, -0.1]:
                new_alpha = current_alpha + delta_a
                if new_alpha < 0 or new_alpha > 1:
                    continue
                H_candidate = (1 - new_alpha) * H_path_norm + new_alpha * H_std_norm
                H_candidate = np.where(H_candidate > 0, H_candidate, 1e-9)
                score = _simulate_aco(prizes, distance_matrix, max_len, H_candidate)
                if score > best_score:
                    best_score = score
                    best_h = H_candidate
                    current_alpha = new_alpha
                    break # Restart loop with new center? Or just continue?
                    # Let's just update best and keep searching
            
    if best_h is None:
        # Fallback to standard heuristic
        best_h = H_std
        best_h = np.where(best_h > 0, best_h, 1e-9)
        
    return best_h
