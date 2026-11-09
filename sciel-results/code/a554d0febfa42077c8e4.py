"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _simulate_aco(prizes, weights, heuristic, ants=20, iters=30, capacity=25, seed=42):
    """
    Run a small ACO simulation to evaluate the quality of a heuristic.
    Returns the best total prize found.
    """
    n_items = len(prizes)
    n_constraints = weights.shape[0]
    
    # Ensure heuristic is positive
    eta = np.where(heuristic > 0, heuristic, 1e-9)
    
    # Pheromone is uniform due to global update, so we only need eta for probabilities
    best_prize_overall = 0.0
    
    rng = np.random.RandomState(seed)
    
    for _ in range(iters):
        best_prize_this_iter = 0.0
        
        for _ in range(ants):
            current_weights = np.zeros(n_constraints)
            total_prize = 0.0
            used_items = np.zeros(n_items, dtype=bool)
            
            # Ant construction
            while True:
                # Find items that fit
                remaining_cap = capacity - current_weights
                fits_mask = np.all(weights.T <= remaining_cap, axis=1)
                fits_mask[used_items] = False
                
                if not np.any(fits_mask):
                    break
                
                # Calculate probabilities
                candidate_indices = np.where(fits_mask)[0]
                candidate_eta = eta[candidate_indices]
                
                sum_probs = np.sum(candidate_eta)
                
                if sum_probs == 0:
                    break
                
                probs = candidate_eta / sum_probs
                
                # Select item
                selected_idx = rng.choice(len(candidate_indices), p=probs)
                item_idx = candidate_indices[selected_idx]
                
                # Add item
                total_prize += prizes[item_idx]
                current_weights += weights[:, item_idx]
                used_items[item_idx] = True
            
        # Update best for this iteration
        if total_prize > best_prize_this_iter:
            best_prize_this_iter = total_prize
            
        if best_prize_this_iter > best_prize_overall:
            best_prize_overall = best_prize_this_iter
            
    return best_prize_overall

def heuristics(prizes, weights):
    start_time = time.time()
    deadline = start_time + 280.0
    
    n_items = len(prizes)
    n_constraints = weights.shape[0]
    
    # Precompute weight features
    sum_w = weights.sum(axis=0)
    max_w = weights.max(axis=0)
    mean_w = weights.mean(axis=0)
    min_w = weights.min(axis=0)
    
    # Pareto-optimal identification: an item is Pareto-dominated if another item
    # has >= prize and <= weight in all constraints (strict in at least one)
    # We use a cheaper proxy: rank by prize and by sum_w, then boost items that
    # are high in prize and low in weight.
    
    candidates = []
    
    # Search grid for exponents: eta = prize^a / (sum_w^b * max_w^c)
    for a in [0.5, 1, 1.5]:
        for b in [0, 0.5, 1, 1.5, 2]:
            for c in [0, 0.5, 1]:
                if b == 0 and c == 0:
                    continue
                denom = 1.0
                if b > 0:
                    denom *= (sum_w + 1e-9) ** b
                if c > 0:
                    denom *= (max_w + 1e-9) ** c
                
                num = prizes ** a if a != 1 else prizes.copy()
                h = num / denom
                
                if np.all(np.isfinite(h)) and np.all(h > 0):
                    candidates.append(h)
    
    # Standard heuristics
    candidates.append(prizes / (sum_w + 1e-9))
    candidates.append(prizes / (max_w + 1e-9))
    candidates.append(prizes / (mean_w + 1e-9))
    candidates.append(prizes / ((sum_w + 1e-9) * (max_w + 1e-9)))
    
    # Log-based heuristic
    log_prize = np.log(prizes + 1e-9)
    log_sum_w = np.log(sum_w + 1e-9)
    candidates.append(np.exp(log_prize - log_sum_w))
    
    # Rank-based: items with high prize and low total weight get high scores
    prize_rank = np.argsort(np.argsort(prizes)) / n_items  # 0 to ~1
    weight_rank = np.argsort(np.argsort(sum_w)) / n_items  # 0 to ~1
    rank_heuristic = prize_rank / (weight_rank + 0.1)
    candidates.append(rank_heuristic)
    
    # Combined: geometric mean of ratio and rank
    ratio_h = prizes / (sum_w + 1e-9)
    geo_mean = np.sqrt(ratio_h * rank_heuristic)
    candidates.append(geo_mean)
    
    # Boost Pareto-like items: items in top half prize AND bottom half weight
    prize_median = np.median(prizes)
    weight_median = np.median(sum_w)
    pareto_boost = np.ones(n_items)
    pareto_mask = (prizes >= prize_median) & (sum_w <= weight_median)
    pareto_boost[pareto_mask] = 2.0
    candidates.append((prizes / (sum_w + 1e-9)) * pareto_boost)
    
    # Quick evaluation to filter candidates
    quick_scores = []
    for h in candidates:
        if time.time() > deadline - 60:
            break
        score = _simulate_aco(prizes, weights, h, ants=8, iters=8, seed=42)
        quick_scores.append(score)
    
    if not quick_scores:
        # Fallback: return basic ratio
        h = prizes / (sum_w + 1e-9)
        return np.where(h > 0, h, 1e-9)
    
    # Sort candidates by quick score
    sorted_indices = np.argsort(quick_scores)[::-1]
    
    best_score = -1
    best_heuristic = candidates[0]
    
    # Full evaluation for top candidates
    top_k = min(12, len(sorted_indices))
    for idx in sorted_indices[:top_k]:
        if time.time() > deadline - 15:
            break
        h = candidates[idx]
        score = _simulate_aco(prizes, weights, h, ants=20, iters=30, seed=42)
        if score > best_score:
            best_score = score
            best_heuristic = h
    
    final_heuristic = np.where(best_heuristic > 0, best_heuristic, 1e-9)
    final_heuristic = np.where(np.isfinite(final_heuristic), final_heuristic, 1e-9)
    
    return final_heuristic
# EVOLVE-BLOCK-END
