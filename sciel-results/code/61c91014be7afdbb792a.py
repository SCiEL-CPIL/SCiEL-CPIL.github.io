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
    
    # Random search parameters
    n_samples = 50
    rng_search = np.random.RandomState(123)
    
    # Generate random alpha, beta pairs
    # alpha: 1 to 10
    # beta: 0 to 5
    alphas = rng_search.uniform(1.0, 10.0, n_samples)
    betas = rng_search.uniform(0.0, 5.0, n_samples)
    
    # Include some fixed points from previous knowledge
    alphas = np.append(alphas, [2.0, 3.0, 4.0, 6.0, 8.0])
    betas = np.append(betas, [0.0, 1.0, 2.0, 3.0])
    
    candidates = list(zip(alphas, betas))
    
    # Shuffle to explore
    rng_search.shuffle(candidates)
    
    # Evaluate candidates using fast proxy
    # We need to evaluate on the specific instance passed in.
    # Note: The task says "3 fixed instances", but the function receives `demands`.
    # The evaluator calls `heuristics` for each instance.
    # So we only optimize for the current instance.
    
    for alpha, beta in candidates:
        if _time_budget_remaining() < 10.0:
            break
            
        # Construct heuristic
        # Handle division by zero or negative base if beta is odd? 
        # gaps are >= 0. 1+gap is always > 0.
        # fills are >= 0.
        
        # If alpha is not integer, fills^alpha is fine for fills >= 0.
        # If fills is 0, 0^alpha is 0.
        
        term1 = np.power(fills, alpha)
        term2 = np.power(1.0 + gaps, -beta)
        
        h = term1 * term2
        h = np.where(fits, h, 1e-9)
        h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
        
        # Fast proxy score
        score = _fast_proxy_score(d, capacity, h, n_ants=5, seed=42)
        
        if score < best_score:
            best_score = score
            best_alpha = alpha
            best_beta = beta
            best_heuristic = h.copy()
            
    # If we didn't find anything or ran out of time, use the best known
    if 'best_heuristic' not in locals():
        # Default to squared fill
        alpha = 2.0
        beta = 1.0
        term1 = np.power(fills, alpha)
        term2 = np.power(1.0 + gaps, -beta)
        best_heuristic = term1 * term2
        best_heuristic = np.where(fits, best_heuristic, 1e-9)
        best_heuristic = np.where(np.isfinite(best_heuristic) & (best_heuristic > 0), best_heuristic, 1e-9)
        
    # Verify the best candidate with full simulation?
    # The proxy might be misleading. Let's do a few full simulations with the top few candidates.
    # But we only kept the best one. Let's keep top 5 from proxy.
    
    # Re-evaluate top 5 with full sim?
    # For simplicity and speed, let's just use the best proxy result.
    # The proxy is a greedy version of the first step. It should correlate.
    
    return best_heuristic

# EVOLVE-BLOCK-END
