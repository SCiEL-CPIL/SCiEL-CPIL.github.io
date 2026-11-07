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
            first_idx = rng.integers(n)
            current_bin_load = W[first_idx]
            current_item = first_idx
            used[first_idx] = True
            
            # Repeatedly pick fitting item
            while True:
                # Vectorized fit check
                remaining_capacity = capacity - current_bin_load
                fits_mask = (W <= remaining_capacity) & (~used)
                
                if not np.any(fits_mask):
                    bins.append(current_bin_load)
                    current_bin_load = 0.0
                    current_item = -1
                    remaining = np.where(~used)[0]
                    if len(remaining) == 0:
                        break
                    first_idx = rng.choice(remaining)
                    current_bin_load = W[first_idx]
                    current_item = first_idx
                    used[first_idx] = True
                    continue
                
                P_row = P[current_item, :]
                H_row = H[current_item, :]
                scores = P_row * H_row
                
                # Zero out non-fitting items
                scores[~fits_mask] = 0.0
                
                total_score = scores.sum()
                if total_score <= 0:
                    candidates = np.where(fits_mask)[0]
                    next_idx = rng.choice(candidates)
                else:
                    probs = scores / total_score
                    next_idx = rng.choice(n, p=probs)
                
                used[next_idx] = True
                current_bin_load += W[next_idx]
                current_item = next_idx
            
            if current_bin_load > 0:
                bins.append(current_bin_load)
            
            if len(bins) < min_bins_this_iter:
                min_bins_this_iter = len(bins)
        
        best_bins_per_iter.append(min_bins_this_iter)
        
        P = P * 0.9 + 1.0 / min_bins_this_iter
    
    return np.mean(best_bins_per_iter)

def _generate_heuristic(W, capacity, mode, alpha=1.0, beta=0.0, gamma=0.0, delta=0.0):
    """
    Generate a heuristic matrix based on the specified mode.
    """
    n = len(W)
    pair_sums = W[:, None] + W[None, :]
    fit_mask = pair_sums <= capacity
    gap = capacity - pair_sums
    
    # Compute degree D[j]: number of items k such that W[j] + W[k] <= capacity
    D = np.sum(fit_mask, axis=0)
    D_safe = np.where(D > 0, D, 1)
    inv_D = 1.0 / D_safe

    # Compute max third item that fits after i and j are placed
    # remaining = capacity - (W[i] + W[j]) = gap
    # max_third[i,j] = max { W[k] : W[k] <= gap[i,j] }
    # If no such k exists (gap < min W), max_third is 0.
    # We want to reward when max_third is 0 (bin is full) or very small.
    # So H_completion ~ 1 / (max_third + epsilon)
    
    # Efficiently compute max_third
    # Sort W
    W_sorted = np.sort(W)
    # For each pair (i,j), gap is known. We need max W[k] <= gap.
    # We can precompute for all possible gaps? No, gaps are continuous-ish.
    # But W values are integers 20..99. So gap is integer.
    # Max possible gap is 150 - 20 - 20 = 110.
    # We can create a lookup table for max W[k] <= g for g in 0..capacity.
    
    max_w_for_gap = np.zeros(capacity + 1, dtype=float)
    # For each g, find max w in W such that w <= g
    # Since W is small, we can just loop
    # Or vectorize: for each w in W, it is the max for g in [w, next_w)
    # Let's just do a simple loop since capacity is 150
    for g in range(capacity + 1):
        candidates = W[W <= g]
        if len(candidates) > 0:
            max_w_for_gap[g] = np.max(candidates)
        else:
            max_w_for_gap[g] = 0.0

    # gap is float, but W is int. Let's cast gap to int for lookup
    gap_int = np.clip(gap.astype(int), 0, capacity)
    max_third = max_w_for_gap[gap_int]
    
    # H_completion: 1 / (max_third + epsilon)
    # If max_third is 0, H is large.
    # If max_third is large, H is small.
    H_completion = 1.0 / (max_third + 1e-6)
    
    # Other components
    H_tight = 1.0 / (gap + 1e-6)
    H_size = W[None, :]
    H_scarcity = inv_D[None, :]
    
    # Combine
    if mode == 'completion':
        H = H_completion
    elif mode == 'completion_tight':
        H = H_completion * H_tight
    elif mode == 'completion_size':
        H = H_completion * H_size
    elif mode == 'completion_scarcity':
        H = H_completion * H_scarcity
    elif mode == 'completion_alpha':
        H = (alpha * H_completion + beta * H_tight + gamma * H_size + delta * H_scarcity)
    elif mode == 'sharp_tight':
        H = np.where(fit_mask, 1.0 / (gap + 1e-10), 1e-9)
    elif mode == 'power_sharp':
        H = np.where(fit_mask, 1.0 / (gap + 1e-10)**1.5, 1e-9)
    elif mode == 'power_sharp2':
        H = np.where(fit_mask, 1.0 / (gap + 1e-10)**2.0, 1e-9)
    elif mode == 'hybrid_sharp':
        H = np.where(fit_mask, W[None, :] / (gap + 1e-10), 1e-9)
    elif mode == 'largest_fit':
        H = np.where(fit_mask, W[None, :], 1e-9)
    elif mode == 'tight_fit':
        H = np.where(fit_mask, 1.0 / (gap + 1e-6), 1e-9)
    elif mode == 'scarcity':
        H = np.where(fit_mask, inv_D[None, :], 1e-9)
    else:
        H = np.where(fit_mask, W[None, :], 1e-9)
        
    H = np.where(fit_mask, H, 1e-9)
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    return H

def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness for placing item j right after item i.
    """
    start_time = time.time()
    deadline = start_time + 290

    W = np.array(demands, dtype=float)
    n = len(W)

    # Prioritize the new "completion" strategies
    candidates = [
        ('completion', {}),
        ('completion_tight', {}),
        ('completion_size', {}),
        ('completion_scarcity', {}),
        ('sharp_tight', {}),
        ('power_sharp2', {}),
        ('hybrid_sharp', {}),
        ('power_sharp', {}),
        # Some alpha blends
        ('completion_alpha', {'alpha': 1.0, 'beta': 1.0, 'gamma': 0.0, 'delta': 0.0}),
        ('completion_alpha', {'alpha': 1.0, 'beta': 0.0, 'gamma': 1.0, 'delta': 0.0}),
        ('completion_alpha', {'alpha': 0.5, 'beta': 0.5, 'gamma': 0.0, 'delta': 0.0}),
        ('largest_fit', {}),
        ('tight_fit', {}),
    ]

    best_score = float('inf')
    best_H = None

    for mode, params in candidates:
        if time.time() > deadline - 30:
            break

        try:
            H = _generate_heuristic(W, capacity, mode, **params)
            score = _simulate_aco(demands, capacity, H, n_ants=20, n_iterations=30, seed=42)
            if score < best_score:
                best_score = score
                best_H = H
        except Exception:
            continue

    if best_H is None:
        best_H = _generate_heuristic(W, capacity, 'sharp_tight')

    return best_H
