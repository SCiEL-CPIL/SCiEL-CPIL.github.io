# EVOLVE-BLOCK-START
import numpy as np
import time

def _run_aco(prizes, dist, max_len, heuristic, n_ants=20, n_iters=30, seed=0):
    n = len(prizes)
    # Precompute which edges are feasible from each node given remaining budget
    # An ant at node i with current length L can go to j if L + dist[i,j] + dist[j,0] <= max_len
    # We'll simulate ants step by step.
    
    pheromone = np.ones((n, n))
    best_prize_overall = 0.0
    rng = np.random.default_rng(seed)
    
    for it in range(n_iters):
        iter_best = 0.0
        for ant in range(n_ants):
            visited = [0]
            current_len = 0.0
            prize_sum = 0.0
            current_node = 0
            unvisited = set(range(1, n))
            
            while True:
                # Find feasible next nodes
                feasible = []
                for j in unvisited:
                    new_len = current_len + dist[current_node, j]
                    if new_len + dist[j, 0] <= max_len + 1e-12:
                        feasible.append(j)
                
                if not feasible:
                    break
                
                # Compute probabilities
                probs = np.zeros(len(feasible))
                for k, j in enumerate(feasible):
                    probs[k] = pheromone[current_node, j] * heuristic[current_node, j]
                
                total = probs.sum()
                if total <= 0:
                    break
                
                probs /= total
                idx = rng.choice(len(feasible), p=probs)
                j = feasible[idx]
                
                current_len += dist[current_node, j]
                prize_sum += prizes[j]
                visited.append(j)
                unvisited.remove(j)
                current_node = j
            
            if prize_sum > iter_best:
                iter_best = prize_sum
        
        if iter_best > best_prize_overall:
            best_prize_overall = iter_best
        
        # Update pheromone
        pheromone *= 0.9
        pheromone += iter_best / 10.0
    
    return best_prize_overall


def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    n = len(prizes)
    dist = distance_matrix
    
    # Base heuristic: prize/distance
    base_heuristic = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and dist[i, j] > 1e-9:
                base_heuristic[i, j] = (prizes[j] + 1e-9) / (dist[i, j] + 1e-9)
            else:
                base_heuristic[i, j] = 1e-9
    
    # Ensure positive
    base_heuristic = np.maximum(base_heuristic, 1e-9)
    
    # We want to find a better heuristic by scaling/exponent tuning.
    # The heuristic is used as pheromone[i,j] * heuristic[i,j].
    # We can try different transformations: heuristic^alpha, or normalize differently.
    
    # Strategy: Try a few heuristic variants and pick the one that gives best ACO result
    # on a quick subset of ants/iterations to save time.
    
    candidates = []
    
    # Candidate 1: base
    candidates.append(base_heuristic.copy())
    
    # Candidate 2: base^0.5 (soften)
    candidates.append(np.power(base_heuristic, 0.5))
    
    # Candidate 3: base^2 (sharpen)
    candidates.append(np.power(base_heuristic, 2.0))
    
    # Candidate 4: prize only (no distance in denom, but normalized)
    h4 = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                h4[i, j] = prizes[j] + 1e-9
            else:
                h4[i, j] = 1e-9
    h4 = np.maximum(h4, 1e-9)
    candidates.append(h4)
    
    # Candidate 5: prize / (distance^0.5)
    h5 = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and dist[i, j] > 1e-9:
                h5[i, j] = (prizes[j] + 1e-9) / (np.sqrt(dist[i, j]) + 1e-9)
            else:
                h5[i, j] = 1e-9
    h5 = np.maximum(h5, 1e-9)
    candidates.append(h5)
    
    # Candidate 6: (prize / distance) but with distance to home considered
    # Heuristic should favor nodes that are far from home (so we can visit more)
    # Actually, we want high prize and low total cost. 
    # Let's try: prize[j] / (dist[i,j] + dist[j,0])
    h6 = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                denom = dist[i, j] + dist[j, 0]
                if denom > 1e-9:
                    h6[i, j] = (prizes[j] + 1e-9) / denom
                else:
                    h6[i, j] = 1e-9
            else:
                h6[i, j] = 1e-9
    h6 = np.maximum(h6, 1e-9)
    candidates.append(h6)
    
    # Normalize each candidate to have mean 1 (to keep pheromone scale similar)
    for idx in range(len(candidates)):
        c = candidates[idx]
        mean_val = c.mean()
        if mean_val > 0:
            candidates[idx] = c / mean_val
    
    # Quick evaluation: run ACO with 5 ants, 5 iterations for each candidate
    quick_scores = []
    for c in candidates:
        score = _run_aco(prizes, dist, max_len, c, n_ants=5, n_iters=5, seed=42)
        quick_scores.append(score)
    
    # Pick the best candidate
    best_idx = int(np.argmax(quick_scores))
    best_heuristic = candidates[best_idx]
    
    # Now, try to further optimize the exponent of the best candidate
    # If the best is base_heuristic, try different exponents
    base_for_tuning = base_heuristic.copy()
    mean_base = base_for_tuning.mean()
    if mean_base > 0:
        base_for_tuning = base_for_tuning / mean_base
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        # Try tuning exponent on the best heuristic family
        # Determine which family is best
        # For simplicity, just tune exponent on base_heuristic if it was among top 2
        alphas = [0.5, 1.0, 1.5, 2.0, 3.0]
        best_alpha_score = quick_scores[best_idx]
        best_alpha = 1.0
        
        for alpha in alphas:
            if time.time() > deadline - 20:
                break
            h_tuned = np.power(base_for_tuning, alpha)
            h_tuned = np.maximum(h_tuned, 1e-9)
            mean_t = h_tuned.mean()
            if mean_t > 0:
                h_tuned = h_tuned / mean_t
            score = _run_aco(prizes, dist, max_len, h_tuned, n_ants=5, n_iters=5, seed=42)
            if score > best_alpha_score:
                best_alpha_score = score
                best_alpha = alpha
        
        if best_alpha != 1.0:
            final_heuristic = np.power(base_for_tuning, best_alpha)
            final_heuristic = np.maximum(final_heuristic, 1e-9)
            mean_f = final_heuristic.mean()
            if mean_f > 0:
                final_heuristic = final_heuristic / mean_f
        else:
            final_heuristic = best_heuristic
    else:
        final_heuristic = best_heuristic
    
    # Final safety check
    final_heuristic = np.maximum(final_heuristic, 1e-9)
    
    return final_heuristic
# EVOLVE-BLOCK-END
