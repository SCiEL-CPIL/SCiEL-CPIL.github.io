import numpy as np
import time

def _simulate_aco(demands, capacity, H, n_ants=20, n_iterations=30, seed=42):
    """
    Simulate the ACO process to evaluate the heuristic matrix H.
    Returns the mean number of bins used by the best ant in each iteration.
    """
    W = np.array(demands, dtype=float)
    n = len(W)
    
    rng = np.random.default_rng(seed)
    
    # Pheromone initialization
    P = np.ones((n, n))
    
    best_bins_per_iter = []
    
    for it in range(n_iterations):
        min_bins_this_iter = float('inf')
        
        for ant in range(n_ants):
            bins = []
            current_bin_load = 0.0
            current_item = -1
            used = np.zeros(n, dtype=bool)
            
            if n == 0:
                continue
            
            # Place first item randomly
            # Any item fits in an empty bin.
            # Random choice.
            first_idx = rng.integers(n)
            current_bin_load = W[first_idx]
            current_item = first_idx
            used[first_idx] = True
            
            # Repeatedly pick fitting item
            while True:
                # Find fitting items
                # An item j fits if current_bin_load + W[j] <= capacity
                # And j is not already used.
                # Vectorized check
                fits_mask = (current_bin_load + W <= capacity) & (~used)
                
                if not np.any(fits_mask):
                    # Open new bin
                    bins.append(current_bin_load)
                    # Reset
                    current_bin_load = 0.0
                    current_item = -1
                    # Pick a random fitting item for the new bin
                    # All items are fitting in an empty bin.
                    # But we must pick an unused one.
                    remaining = np.where(~used)[0]
                    if len(remaining) == 0:
                        break
                    first_idx = rng.choice(remaining)
                    current_bin_load = W[first_idx]
                    current_item = first_idx
                    used[first_idx] = True
                    continue
                
                # Calculate probabilities
                # P[current_item, j] * H[current_item, j]
                P_row = P[current_item, :]
                H_row = H[current_item, :]
                scores = P_row * H_row
                
                # Mask out non-fitting items
                scores = np.where(fits_mask, scores, 0.0)
                
                total_score = scores.sum()
                if total_score == 0:
                    # Fallback: pick random fitting item
                    candidates = np.where(fits_mask)[0]
                    next_idx = rng.choice(candidates)
                else:
                    probs = scores / total_score
                    # Sample
                    next_idx = rng.choice(n, p=probs)
                
                used[next_idx] = True
                current_bin_load += W[next_idx]
                current_item = next_idx
            
            # Finalize last bin
            if current_bin_load > 0:
                bins.append(current_bin_load)
            
            if len(bins) < min_bins_this_iter:
                min_bins_this_iter = len(bins)
        
        best_bins_per_iter.append(min_bins_this_iter)
        
        # Pheromone update
        # "pheromone is multiplied by 0.9 and every entry gains 1/(fewest bins of that iteration)"
        P = P * 0.9 + 1.0 / min_bins_this_iter
    
    return np.mean(best_bins_per_iter)

def _generate_heuristic(W, capacity, mode):
    """
    Generate a heuristic matrix based on the specified mode.
    """
    n = len(W)
    pair_sums = W[:, None] + W[None, :]
    fit_mask = pair_sums <= capacity
    
    if mode == 'largest_fit':
        # Prefer larger items j
        H = np.where(fit_mask, W[None, :], 1e-9)
        
    elif mode == 'tight_fit':
        # Prefer items j that make w_i + w_j close to capacity
        # H[i,j] = 1 / (capacity - (w_i + w_j))
        # If capacity - sum is 0, it's a perfect fit, very attractive.
        # Use a small epsilon to avoid division by zero.
        gap = capacity - pair_sums
        # For non-fitting, gap > 0, but we mask them out.
        # For fitting, gap >= 0.
        # We want large H for small gap.
        # H = 1 / (gap + 1e-6)
        H = np.where(fit_mask, 1.0 / (gap + 1e-6), 1e-9)
        
    elif mode == 'hybrid':
        # Combine largest fit and tight fit
        # H = w_j / (gap + 1)
        gap = capacity - pair_sums
        H = np.where(fit_mask, W[None, :] / (gap + 1.0), 1e-9)
        
    elif mode == 'normalized_tight':
        gap = capacity - pair_sums
        H_raw = np.where(fit_mask, 1.0 / (gap + 1e-6), 0.0)
        # Normalize to mean 1 for fitting entries
        fitting_vals = H_raw[fit_mask]
        if len(fitting_vals) > 0:
            mean_val = np.mean(fitting_vals)
            if mean_val > 0:
                H = H_raw / mean_val
            else:
                H = H_raw
        else:
            H = H_raw
        H = np.where(H > 0, H, 1e-9)
            
    else:
        # Default
        H = np.where(fit_mask, W[None, :], 1e-9)
        
    # Ensure positivity and finiteness
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    return H

def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness for placing item j right after item i.
    """
    start_time = time.time()
    deadline = start_time + 290  # 290s limit, keep 10s margin for safety
    
    W = np.array(demands, dtype=float)
    n = len(W)
    
    modes = ['largest_fit', 'tight_fit', 'hybrid', 'normalized_tight']
    
    best_score = float('inf')
    best_H = None
    
    for mode in modes:
        if time.time() > deadline - 30:  # Stop if not enough time left for evaluation
            break
            
        H = _generate_heuristic(W, capacity, mode)
        
        try:
            score = _simulate_aco(demands, capacity, H, n_ants=20, n_iterations=30, seed=42)
            if score < best_score:
                best_score = score
                best_H = H
        except Exception:
            continue
            
    if best_H is None:
        # Fallback if all failed or no time
        best_H = _generate_heuristic(W, capacity, 'largest_fit')
        
    return best_H
