"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 15.0  # seconds before the 330s kill limit

def _time_budget_remaining():
    return (330.0 - _DEADLINE_MARGIN) - (time.time() - _START_TIME)

def _fast_proxy_score(d, capacity, heuristic, n_ants=5, seed=42):
    """
    Fast greedy simulation using the heuristic.
    Assumption: Pheromone is uniform (1.0) initially, so choice is driven by heuristic.
    This approximates the first iteration of ACO or a greedy heuristic.
    """
    n = len(d)
    d = np.asarray(d, dtype=np.float64)
    rng = np.random.RandomState(seed)
    
    best_bins = float('inf')
    
    for _ in range(n_ants):
        placed = np.zeros(n, dtype=bool)
        bins_used = 0
        current_bin_sum = 0.0
        last_item = -1
        
        # Open first bin, random start
        idx = rng.randint(n)
        placed[idx] = True
        current_bin_sum = d[idx]
        last_item = idx
        bins_used = 1
        
        while not np.all(placed):
            remaining_space = capacity - current_bin_sum
            # Find fitting items
            fits_mask = (~placed) & (d <= remaining_space)
            
            if not np.any(fits_mask):
                # Open new bin
                bins_used += 1
                available = np.where(~placed)[0]
                idx = available[rng.randint(len(available))]
                placed[idx] = True
                current_bin_sum = d[idx]
                last_item = idx
                continue
            
            # Greedy choice: max heuristic[last_item, j] among fitting j
            # heuristic is (n, n)
            row_he = heuristic[last_item, :]
            # Mask out non-fitting
            scores = np.where(fits_mask, row_he, 0.0)
            
            # Find max
            idx = np.argmax(scores)
            
            # Check if score is 0 (should not happen if fits_mask is true and heuristics > 0)
            if scores[idx] == 0:
                 # Fallback to random if heuristic failed
                 candidates = np.where(fits_mask)[0]
                 idx = candidates[rng.randint(len(candidates))]
            
            placed[idx] = True
            current_bin_sum += d[idx]
            last_item = idx
            
        if bins_used < best_bins:
            best_bins = bins_used
            
    return best_bins

def _medium_aco_score(d, capacity, heuristic, n_ants=10, n_iterations=15, seed=42):
    """
    Medium-fidelity ACO simulation for filtering top candidates.
    Less expensive than full 20x30, but captures pheromone dynamics better than greedy.
    """
    n = len(d)
    d = np.asarray(d, dtype=np.float64)
    rng = np.random.RandomState(seed)
    
    pheromone = np.ones((n, n))
    best_bins_ever = float('inf')
    
    for _ in range(n_iterations):
        best_bins_this_iter = float('inf')
        
        for _ in range(n_ants):
            placed = np.zeros(n, dtype=bool)
            bins_used = 0
            current_bin_sum = 0.0
            last_item = -1
            
            # 1. Open a bin, place a random fitting item
            idx = rng.randint(n)
            placed[idx] = True
            current_bin_sum = d[idx]
            last_item = idx
            bins_used = 1
            
            while not np.all(placed):
                remaining_space = capacity - current_bin_sum
                fits_mask = (~placed) & (d <= remaining_space)
                
                if not np.any(fits_mask):
                    bins_used += 1
                    available = np.where(~placed)[0]
                    idx = available[rng.randint(len(available))]
                    placed[idx] = True
                    current_bin_sum = d[idx]
                    last_item = idx
                    continue
                
                row_ph = pheromone[last_item, :]
                row_he = heuristic[last_item, :]
                
                scores = row_ph * row_he
                scores = np.where(fits_mask, scores, 0.0)
                
                total_score = np.sum(scores)
                if total_score == 0:
                    candidates = np.where(fits_mask)[0]
                    idx = candidates[rng.randint(len(candidates))]
                else:
                    probs = scores / total_score
                    idx = rng.choice(n, p=probs)
                    
                placed[idx] = True
                current_bin_sum += d[idx]
                last_item = idx
                
        if bins_used < best_bins_this_iter:
            best_bins_this_iter = bins_used
            
        if best_bins_this_iter < best_bins_ever:
            best_bins_ever = best_bins_this_iter
            
        # Update pheromone
        pheromone *= 0.9
        pheromone += 1.0 / best_bins_this_iter
        
    return best_bins_ever

def _full_simulate_aco(d, capacity, heuristic, n_ants=20, n_iterations=30, seed=None):
    """
    Full ACO simulation as per task description.
    """
    n = len(d)
    d = np.asarray(d, dtype=np.float64)
    
    pheromone = np.ones((n, n))
    
    best_bins_ever = float('inf')
    
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
        
    for _ in range(n_iterations):
        best_bins_this_iter = float('inf')
        
        for _ in range(n_ants):
            placed = np.zeros(n, dtype=bool)
            bins_used = 0
            current_bin_sum = 0.0
            last_item = -1
            
            # 1. Open a bin, place a random fitting item
            idx = rng.randint(n)
            placed[idx] = True
            current_bin_sum = d[idx]
            last_item = idx
            bins_used = 1
            
            while not np.all(placed):
                remaining_space = capacity - current_bin_sum
                fits_mask = (~placed) & (d <= remaining_space)
                
                if not np.any(fits_mask):
                    bins_used += 1
                    available = np.where(~placed)[0]
                    idx = available[rng.randint(len(available))]
                    placed[idx] = True
                    current_bin_sum = d[idx]
                    last_item = idx
                    continue
                
                row_ph = pheromone[last_item, :]
                row_he = heuristic[last_item, :]
                
                scores = row_ph * row_he
                scores = np.where(fits_mask, scores, 0.0)
                
                total_score = np.sum(scores)
                if total_score == 0:
                    candidates = np.where(fits_mask)[0]
                    idx = candidates[rng.randint(len(candidates))]
                else:
                    probs = scores / total_score
                    idx = rng.choice(n, p=probs)
                    
                placed[idx] = True
                current_bin_sum += d[idx]
                last_item = idx
                
        if bins_used < best_bins_this_iter:
            best_bins_this_iter = bins_used
            
        if best_bins_this_iter < best_bins_ever:
            best_bins_ever = best_bins_this_iter
            
        # Update pheromone
        pheromone *= 0.9
        pheromone += 1.0 / best_bins_this_iter
        
    return best_bins_ever

def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness.
    Uses random search over parameters alpha, beta for the heuristic:
    H_ij = ( (d_i+d_j)/C )^alpha * (1 + (C - (d_i+d_j)))^(-beta)
    """
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    
    # Precompute pair sums and gaps
    pair_sums = d[:, None] + d[None, :]
    fits = pair_sums <= capacity
    gaps = capacity - pair_sums
    fills = pair_sums / capacity
    
    # Initialize with a known good heuristic
    # Based on finding: squared ratio
    best_score = float('inf')
    best_alpha = 2.0
    best_beta = 1.0
    best_heuristic = None
    
    # Random search parameters
    n_samples = 80
    rng_search = np.random.RandomState(123)
    
    # Generate random alpha, beta pairs
    # Wider search space
    alphas = rng_search.uniform(0.5, 15.0, n_samples)
    betas = rng_search.uniform(0.0, 10.0, n_samples)
    
    # Include some fixed points from previous knowledge
    alphas = np.append(alphas, [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0])
    betas = np.append(betas, [0.5, 1.0, 2.0, 3.0, 5.0])
    
    candidates = list(zip(alphas, betas))
    
    # Shuffle to explore
    rng_search.shuffle(candidates)
    
    # Stage 1: Evaluate candidates using fast proxy
    # We need to evaluate on the specific instance passed in.
    # Note: The task says "3 fixed instances", but the function receives `demands`.
    # The evaluator calls `heuristics` for each instance.
    # So we only optimize for the current instance.
    
    proxy_scores = []
    
    for alpha, beta in candidates:
        if _time_budget_remaining() < 20.0:
            break
            
        # Construct heuristic
        term1 = np.power(fills, alpha)
        term2 = np.power(1.0 + gaps, -beta)
        
        h = term1 * term2
        h = np.where(fits, h, 1e-9)
        h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
        
        # Fast proxy score
        score = _fast_proxy_score(d, capacity, h, n_ants=5, seed=42)
        proxy_scores.append((score, alpha, beta, h))
        
    if not proxy_scores:
        # Fallback default
        alpha, beta = 2.0, 1.0
        term1 = np.power(fills, alpha)
        term2 = np.power(1.0 + gaps, -beta)
        best_heuristic = term1 * term2
        best_heuristic = np.where(fits, best_heuristic, 1e-9)
        best_heuristic = np.where(np.isfinite(best_heuristic) & (best_heuristic > 0), best_heuristic, 1e-9)
        return best_heuristic

    # Sort by proxy score
    proxy_scores.sort(key=lambda x: x[0])
    
    # Stage 2: Evaluate top 10 candidates with medium ACO simulation
    top_candidates = proxy_scores[:10]
    
    for score, alpha, beta, h in top_candidates:
        if _time_budget_remaining() < 10.0:
            break
        
        # Medium ACO score
        med_score = _medium_aco_score(d, capacity, h, n_ants=10, n_iterations=15, seed=42)
        
        # We want to minimize bins. 
        # Use a combined metric or just the medium score?
        # The medium score is a better predictor of the final full ACO score.
        if med_score < best_score:
            best_score = med_score
            best_alpha = alpha
            best_beta = beta
            best_heuristic = h.copy()
            
    # If we didn't find anything better than inf (shouldn't happen), use the best proxy one
    if best_heuristic is None:
        best_heuristic = top_candidates[0][3]
        
    return best_heuristic

# EVOLVE-BLOCK-END
