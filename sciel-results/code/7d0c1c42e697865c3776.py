import numpy as np
import time

def _simulate_aco(prizes, distance_matrix, max_len, heuristic, num_ants=20, num_iters=30, seed=42):
    n = len(prizes)
    H = np.where(heuristic > 0, heuristic, 1e-9)
    
    best_prize = 0
    rng = np.random.default_rng(seed)
    
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
                # d[current_node, :] + d[:, 0] <= remaining_budget
                valid_indices = np.where(
                    (~visited_mask) & 
                    (distance_matrix[current_node, :] + distance_matrix[:, 0] <= remaining_budget)
                )[0]
                
                if len(valid_indices) == 0:
                    break
                
                # Get heuristic weights for valid nodes
                h_vals = H[current_node, valid_indices]
                
                # Normalize to probabilities
                total_h = np.sum(h_vals)
                if total_h == 0:
                    break
                    
                probs = h_vals / total_h
                
                # Sample next node
                next_node = valid_indices[rng.choice(len(valid_indices), p=probs)]
                
                # Update state
                path_len += distance_matrix[current_node, next_node]
                total_prize += prizes[next_node]
                visited_mask[next_node] = True
                current_node = next_node
            
        best_iter_prize = max(best_iter_prize, total_prize)
        best_prize = max(best_prize, best_iter_prize)

    return best_prize

def _greedy_path(prizes, distance_matrix, max_len, mode='prize_density'):
    """
    Constructs a greedy path starting at 0.
    mode: 'prize_density' (prize/distance), 'prize' (max prize), 'slack' (max remaining slack)
    """
    n = len(prizes)
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    current = 0
    path_len = 0.0
    path = [0]
    
    while True:
        remaining = max_len - path_len
        valid = []
        scores = []
        
        for j in range(n):
            if visited[j]:
                continue
            d_ij = distance_matrix[current, j]
            d_j0 = distance_matrix[j, 0]
            if d_ij + d_j0 <= remaining:
                valid.append(j)
                if mode == 'prize_density':
                    d_eff = max(d_ij, 1e-6)
                    scores.append(prizes[j] / d_eff)
                elif mode == 'prize':
                    scores.append(prizes[j])
                elif mode == 'slack':
                    # Prefer nodes that leave more slack? Or just take best prize among those that fit?
                    # Actually, for 'slack', we might want to maximize remaining budget after move.
                    # But usually greedy takes best value. Let's stick to prize-based for simplicity in greedy.
                    # Let's use prize/distance for this mode too, but maybe different tie-breaking?
                    # Or just max prize.
                    scores.append(prizes[j])
        
        if not valid:
            break
        
        # Find best
        best_idx = np.argmax(scores)
        next_node = valid[best_idx]
        
        path.append(next_node)
        visited[next_node] = True
        path_len += distance_matrix[current, next_node]
        current = next_node
        
    return path

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    n = len(prizes)
    eps = 1e-9
    
    # 1. Generate candidate paths using different greedy strategies
    # We want to find edges that are part of high-quality routes.
    # We will combine heuristics from multiple good paths.
    
    candidate_paths = []
    
    # Greedy by Prize/Density
    p1 = _greedy_path(prizes, distance_matrix, max_len, mode='prize_density')
    candidate_paths.append(p1)
    
    # Greedy by Prize (Max Prize first)
    p2 = _greedy_path(prizes, distance_matrix, max_len, mode='prize')
    candidate_paths.append(p2)
    
    # Greedy by Distance (Nearest Neighbor) - might find dense clusters
    # Let's implement a simple NN
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    current = 0
    path_len = 0.0
    p3 = [0]
    while True:
        remaining = max_len - path_len
        valid = []
        dists = []
        for j in range(n):
            if visited[j]:
                continue
            d_ij = distance_matrix[current, j]
            d_j0 = distance_matrix[j, 0]
            if d_ij + d_j0 <= remaining:
                valid.append(j)
                dists.append(d_ij)
        if not valid:
            break
        best_idx = np.argmin(dists)
        next_node = valid[best_idx]
        p3.append(next_node)
        visited[next_node] = True
        path_len += distance_matrix[current, next_node]
        current = next_node
    candidate_paths.append(p3)
    
    # 2. Construct the "Regret-Boosted" Heuristic
    # Base heuristic: Prize / (Distance + small_offset)
    # This provides a general gradient towards high prize, low cost edges.
    
    # We want to boost edges that appear in the candidate paths.
    # Let's create a base matrix H_base.
    
    # Offset to avoid division by zero and smooth the matrix
    dist_offset = 0.05
    d_mat = distance_matrix + dist_offset
    p_vec = prizes + eps
    
    # H_base[i, j] = prizes[j] / d_mat[i, j]
    # Note: In ACO, heuristic[i, j] is used when moving from i to j.
    # So we care about prize at j.
    
    H_base = p_vec[None, :] / d_mat
    
    # Ensure positive
    H_base = np.maximum(H_base, eps)
    
    # Now, determine the boost.
    # We will sum up contributions from each candidate path.
    # For each edge (i, j) in a path, we add a boost.
    # The boost should be significant enough to guide the ants, but not so dominant
    # that it completely ignores the pheromone or other options if the path is suboptimal.
    
    # Let's calculate the total prize of each path to weight the boost?
    # Or just a fixed boost?
    # A fixed boost on "good" edges is like pre-seeding pheromone.
    
    # Let's try a boost proportional to the prize of the destination node,
    # scaled by a factor.
    
    H_final = H_base.copy()
    
    for path in candidate_paths:
        if len(path) < 2:
            continue
        for k in range(len(path) - 1):
            i = path[k]
            j = path[k+1]
            # Boost edge i->j
            # Boost amount: proportional to prizes[j]
            # Let's use a multiplier, e.g., 5x the base heuristic?
            # Or add a constant?
            # Adding to H_base is safer.
            # Let's add prizes[j] * 2.0 to H[i, j]
            # This makes the "greedy choice" very attractive.
            H_final[i, j] += prizes[j] * 2.0
            
            # Also boost reverse edge?
            # In Orienteering, path direction matters.
            # But ACO uses H[i,j] for i->j.
            # If the greedy path went i->j, it's good.
            # Does j->i matter? Only if an ant is at j and wants to go to i.
            # But i is visited in the greedy path.
            # However, other ants might visit j before i.
            # So boosting j->i might also be useful if it leads to a good continuation.
            # Let's boost both for symmetry in the "attractiveness" of the pair,
            # but primarily the forward edge.
            H_final[j, i] += prizes[i] * 0.5 # Smaller boost for reverse
    
    # 3. Refinement: Local Search on the Heuristic?
    # The current approach is non-parametric.
    # Can we improve it?
    # Maybe try a few variations of the boost strength?
    
    best_score = -1
    best_h = H_final
    
    # Let's evaluate the current H_final
    if time.time() < deadline:
        score = _simulate_aco(prizes, distance_matrix, max_len, H_final)
        if score > best_score:
            best_score = score
            best_h = H_final
    
    # Try varying the boost strength
    # We stored the base and the boosts implicitly.
    # Let's re-generate with different multipliers.
    
    multipliers = [1.0, 2.0, 5.0, 10.0]
    
    for mult in multipliers:
        if time.time() > deadline - 10:
            break
            
        H_test = H_base.copy()
        for path in candidate_paths:
            if len(path) < 2:
                continue
            for k in range(len(path) - 1):
                i = path[k]
                j = path[k+1]
                H_test[i, j] += prizes[j] * mult
                H_test[j, i] += prizes[i] * (mult * 0.5)
        
        # Ensure positive
        H_test = np.maximum(H_test, eps)
        
        score = _simulate_aco(prizes, distance_matrix, max_len, H_test)
        if score > best_score:
            best_score = score
            best_h = H_test
    
    # Try a "Pure Regret" heuristic?
    # Regret[i, j] = H_base[i, best_j] - H_base[i, j]
    # High regret means "I really want to go to best_j, not j".
    # So we want to boost best_j.
    # This is similar to what we did, but calculated dynamically per row.
    
    if time.time() < deadline - 10:
        H_regret = H_base.copy()
        for i in range(n):
            if i == 0:
                continue # Start node, no incoming edges needed for heuristic? 
                         # Actually H[0, j] is used for first move.
            
            # Find best j for row i based on H_base
            # But we should only consider valid moves? 
            # Heuristic is static, so it doesn't know about visited/budget.
            # So we just rank by H_base.
            
            # Get sorted indices for row i
            # Exclude i itself? Distance matrix diagonal is 0, H is high.
            # Usually we don't move to self.
            # Let's assume distance_matrix[i, i] = 0, so H_base[i, i] is high.
            # We should probably mask self.
            
            row = H_base[i].copy()
            row[i] = 0 # Don't go to self
            
            best_j = np.argmax(row)
            second_best_val = 0
            # Get second best
            row[best_j] = 0
            second_best_j = np.argmax(row)
            
            # Regret for best_j is 0 (it's the best).
            # Regret for others is (Best - Current).
            # We want to boost the best_j.
            # So H_regret[i, best_j] += large_value?
            
            # Actually, the "Regret" heuristic in ACO often means:
            # If I don't take the best option, I lose out.
            # So we want to make the best option VERY attractive.
            
            # Let's add a boost to the best_j in each row.
            H_regret[i, best_j] += prizes[best_j] * 3.0
        
        H_regret = np.maximum(H_regret, eps)
        
        score = _simulate_aco(prizes, distance_matrix, max_len, H_regret)
        if score > best_score:
            best_score = score
            best_h = H_regret
    
    # Fallback
    if best_h is None:
        best_h = H_base
    
    # Final check for validity
    if best_h.shape != (n, n) or not np.all(np.isfinite(best_h)):
        best_h = H_base
        best_h = np.where(best_h > 0, best_h, 1e-9)
        
    return best_h
