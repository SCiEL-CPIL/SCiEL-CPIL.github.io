import numpy as np
import time

def _simulate_aco(demands, capacity, H, n_ants=20, n_iterations=30, seed=42):
    """
    Simulate the ACO process to evaluate the heuristic matrix H.
    Returns the mean number of bins used by the best ant in each iteration.
    """
    W = np.array(demands, dtype=float)
    n = len(W)
    
    if n == 0:
        return 0.0

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
            
            # Place first item randomly
            first_idx = rng.integers(n)
            current_bin_load = W[first_idx]
            current_item = first_idx
            used[first_idx] = True
            
            # Repeatedly pick fitting item
            while True:
                fits_mask = (current_bin_load + W <= capacity) & (~used)
                
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
                
                scores = np.where(fits_mask, scores, 0.0)
                
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

def _generate_heuristic(W, capacity, mode):
    """
    Generate a heuristic matrix based on the specified mode.
    """
    n = len(W)
    pair_sums = W[:, None] + W[None, :]
    fit_mask = pair_sums <= capacity
    gap = capacity - pair_sums
    
    if mode == 'sharp_tight':
        # Pure 1/gap: strongest preference for near-perfect fits
        H = np.where(fit_mask, 1.0 / (gap + 1e-10), 1e-9)
        
    elif mode == 'power_sharp':
        # 1/gap^1.5: even steeper preference
        H = np.where(fit_mask, 1.0 / (gap + 1e-10)**1.5, 1e-9)
        
    elif mode == 'power_sharp2':
        # 1/gap^2: steepest preference
        H = np.where(fit_mask, 1.0 / (gap + 1e-10)**2.0, 1e-9)
        
    elif mode == 'hybrid_sharp':
        # w_j / gap: combine size preference with tightness
        H = np.where(fit_mask, W[None, :] / (gap + 1e-10), 1e-9)
        
    elif mode == 'largest_fit':
        H = np.where(fit_mask, W[None, :], 1e-9)
        
    elif mode == 'tight_fit':
        H = np.where(fit_mask, 1.0 / (gap + 1e-6), 1e-9)
        
    elif mode == 'scarcity':
        # Scarcity-based: items with fewer compatible partners are more valuable
        # For each item j, count how many items i can fit with j
        # degree[j] = number of items that can fit with j
        # Heuristic: higher for items with lower degree (scarcer partners)
        # H[i,j] = 1 / (degree[j] + 1) * (1/gap)
        compat = pair_sums <= capacity
        degree = compat.sum(axis=0)  # number of items that can fit with each j
        # Inverse degree: scarce items get higher score
        inv_degree = 1.0 / (degree + 1.0)
        H = np.where(fit_mask, inv_degree[None, :] / (gap + 1e-10), 1e-9)
        
    elif mode == 'scarcity_size':
        # Combine scarcity with size
        compat = pair_sums <= capacity
        degree = compat.sum(axis=0)
        inv_degree = 1.0 / (degree + 1.0)
        H = np.where(fit_mask, inv_degree[None, :] * W[None, :] / (gap + 1e-10), 1e-9)
        
    elif mode == 'centrality':
        # PageRank-like centrality in compatibility graph
        compat = pair_sums <= capacity
        # Simple power iteration for centrality
        centrality = np.ones(n) / n
        for _ in range(20):
            new_cent = np.zeros(n)
            for j in range(n):
                neighbors = np.where(compat[:, j])[0]
                if len(neighbors) > 0:
                    new_cent[j] = np.sum(centrality[neighbors]) / len(neighbors)
            centrality = new_cent
        H = np.where(fit_mask, centrality[None, :] / (gap + 1e-10), 1e-9)
        
    else:
        H = np.where(fit_mask, W[None, :], 1e-9)
        
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

    # Order candidates by expected quality: include scarcity-based heuristics
    modes = ['sharp_tight', 'scarcity', 'scarcity_size', 'power_sharp', 'hybrid_sharp', 'centrality', 'power_sharp2', 'tight_fit', 'largest_fit']

    best_score = float('inf')
    best_H = None

    for mode in modes:
        if time.time() > deadline - 40:
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
        best_H = _generate_heuristic(W, capacity, 'sharp_tight')

    return best_H
