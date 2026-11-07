"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _time_budget_remaining(limit=300, safety_margin=5):
    """Check if we have enough time left to run the simulation."""
    # This is a placeholder. The actual time tracking is done in heuristics.
    return True

def _fast_ffd_proxy(demands, capacity, heuristic):
    """
    A very fast proxy for ACO performance.
    Since ACO is stochastic and slow, we use a greedy First-Fit-Decreasing (FFD)
    variant guided by the heuristic as a tie-breaker or primary sort key.
    
    Actually, a better proxy for "how good is this heuristic" is to simulate 
    a simplified ACO or just use the heuristic to guide a greedy packing.
    However, the task asks for a structural change. 
    Let's use a "Guided First Fit" where we sort items by size desc, 
    but when placing, we pick the bin that minimizes the resulting waste 
    according to the heuristic potential.
    
    For speed in the optimization loop, we'll use a simplified simulation:
    1. Sort items descending.
    2. For each item, try to place in existing bins.
    3. If multiple bins fit, pick the one that maximizes heuristic[last_in_bin, item].
    4. If none fit, open new bin.
    
    This is O(n^2) per instance, which is fast enough for many iterations.
    """
    n = len(demands)
    # Sort indices by demand descending
    sorted_idx = np.argsort(demands)[::-1]
    
    bins = []  # list of lists of item indices
    bin_sums = []
    
    for idx in sorted_idx:
        item_size = demands[idx]
        best_bin_idx = -1
        best_heuristic_val = -1e9
        
        for b_idx, b in enumerate(bins):
            if bin_sums[b_idx] + item_size <= capacity:
                if len(b) > 0:
                    last_item = b[-1]
                    h_val = heuristic[last_item, idx]
                    if h_val > best_heuristic_val:
                        best_heuristic_val = h_val
                        best_bin_idx = b_idx
                else:
                    # Empty bin: heuristic is less relevant, but let's say 0
                    # Or we can just prefer empty bins if no other choice
                    if best_bin_idx == -1:
                        best_bin_idx = b_idx
                        best_heuristic_val = 0 # Neutral
        
        if best_bin_idx != -1:
            bins[best_bin_idx].append(idx)
            bin_sums[best_bin_idx] += item_size
        else:
            bins.append([idx])
            bin_sums.append(item_size)
            
    return len(bins)

def _compute_heuristic(demands, capacity, alpha, beta, gamma):
    """
    Compute the heuristic matrix based on waste compatibility.
    Score = base_fill^alpha + beta * (1 / (waste + gamma)^2)
    """
    n = len(demands)
    pair = demands[:, None] + demands[None, :]
    fits = pair <= capacity
    waste = capacity - pair
    
    # Base fill ratio
    fill_ratio = np.where(fits, pair / capacity, 0.0)
    base_score = fill_ratio ** alpha
    
    # Waste compatibility: strongly penalize large waste
    # If waste is 0, 1/(0+gamma)^2 is huge.
    # If waste is large, it's small.
    # We want to REPEL from large waste, so high score for low waste.
    waste_term = np.zeros((n, n))
    valid_waste = fits & (waste >= 0)
    waste_term[valid_waste] = 1.0 / (waste[valid_waste] + gamma) ** 2
    
    matrix = base_score + beta * waste_term
    
    # Mask non-fitting
    matrix = np.where(fits, matrix, 1e-9)
    
    # Ensure diagonal
    for i in range(n):
        if 2.0 * demands[i] <= capacity:
            w = capacity - 2.0 * demands[i]
            fr = (2.0 * demands[i]) / capacity
            score = fr ** alpha + beta * (1.0 / (w + gamma) ** 2)
            matrix[i, i] = score
        else:
            matrix[i, i] = 1e-9
            
    matrix = np.where(np.isfinite(matrix), matrix, 1e-9)
    return matrix

def heuristics(demands, capacity):
    """
    Optimize heuristic parameters (alpha, beta, gamma) using a fast proxy search.
    """
    demands = np.asarray(demands, dtype=np.float64)
    n = len(demands)
    
    start_time = time.time()
    deadline = start_time + 290.0  # 10s safety margin
    
    # Define search space
    # alpha: power of fill ratio (0.5 to 5.0)
    # beta: weight of waste term (0.0 to 5.0)
    # gamma: smoothing factor for waste (0.5 to 10.0)
    
    # Initial best
    best_score = float('inf')
    best_params = (1.0, 1.0, 1.0)
    
    # Random sampling + Local refinement
    np.random.seed(42) # Fixed seed for reproducibility within the function? 
    # The task says "seeded per instance" for ACO, but our heuristic is computed once.
    # We should probably use a deterministic search or a fixed seed to be safe, 
    # but random search with a fixed seed is fine.
    
    # 1. Coarse random search
    num_coarse = 50
    for _ in range(num_coarse):
        if time.time() > deadline:
            break
        alpha = np.random.uniform(0.5, 5.0)
        beta = np.random.uniform(0.0, 5.0)
        gamma = np.random.uniform(0.5, 10.0)
        
        h = _compute_heuristic(demands, capacity, alpha, beta, gamma)
        score = _fast_ffd_proxy(demands, capacity, h)
        
        if score < best_score:
            best_score = score
            best_params = (alpha, beta, gamma)
            
    # 2. Local refinement around best
    # Simple hill climbing / random perturbation
    current_params = best_params
    for step in range(50):
        if time.time() > deadline:
            break
        
        # Perturb
        new_alpha = current_params[0] + np.random.normal(0, 0.5)
        new_beta = current_params[1] + np.random.normal(0, 0.5)
        new_gamma = current_params[2] + np.random.normal(0, 1.0)
        
        # Constrain
        new_alpha = np.clip(new_alpha, 0.1, 10.0)
        new_beta = np.clip(new_beta, 0.0, 10.0)
        new_gamma = np.clip(new_gamma, 0.1, 20.0)
        
        h = _compute_heuristic(demands, capacity, new_alpha, new_beta, new_gamma)
        score = _fast_ffd_proxy(demands, capacity, h)
        
        if score < best_score:
            best_score = score
            best_params = (new_alpha, new_beta, new_gamma)
            current_params = best_params
        else:
            # If no improvement, shrink step size? Or just continue.
            # For simplicity, just continue.
            pass
            
    # Compute final heuristic with best params
    final_h = _compute_heuristic(demands, capacity, *best_params)
    
    # Final safety check
    if not np.all(np.isfinite(final_h)):
        # Fallback to a safe default
        pair = demands[:, None] + demands[None, :]
        fits = pair <= capacity
        waste = capacity - pair
        fill_ratio = np.where(fits, pair / capacity, 0.0)
        base_score = fill_ratio ** 2
        waste_term = np.zeros((n, n))
        valid_waste = fits
        waste_term[valid_waste] = 1.0 / (waste[valid_waste] + 1.0) ** 2
        final_h = base_score + 1.0 * waste_term
        final_h = np.where(fits, final_h, 1e-9)
        final_h = np.where(np.isfinite(final_h), final_h, 1e-9)
        
    return final_h
# EVOLVE-BLOCK-END
