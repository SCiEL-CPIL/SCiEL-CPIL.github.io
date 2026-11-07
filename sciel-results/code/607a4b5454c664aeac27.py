import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ACO bin packing algorithm.
    Uses internal ACO simulation to find the best heuristic parameters.
    """
    W = np.array(demands, dtype=float)
    n = len(W)
    
    start_time = time.time()
    deadline = start_time + 250  # Safety margin
    
    # Precompute common terms
    pair_sums = W[:, None] + W[None, :]
    gaps = capacity - pair_sums
    fits = gaps >= 0
    eps = 1e-9
    safe_gaps = np.where(fits, gaps + eps, 1.0)
    fill_ratio = np.where(fits, pair_sums / capacity, 0.0)
    item_size = W[None, :]
    
    # Candidate parameters: (alpha_fill, gap_power)
    # alpha_fill: weight for fill-ratio bonus
    # gap_power: exponent on gap in denominator
    candidates = []
    for alpha_fill in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        for gap_power in [0.5, 1.0, 1.5, 2.0]:
            candidates.append((alpha_fill, gap_power))
    
    best_H = None
    best_score = float('inf')
    
    # Evaluate each candidate
    for alpha_fill, gap_power in candidates:
        if time.time() > deadline:
            break
        
        # Compute heuristic
        base_term = item_size / (safe_gaps ** gap_power)
        fill_bonus = 1.0 + alpha_fill * (fill_ratio ** 2)
        H_valid = base_term * fill_bonus
        H = np.where(fits, H_valid, 1e-9)
        H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
        
        # Quick evaluation with a few seeds
        scores = []
        for seed in [42, 123, 999]:
            if time.time() > deadline:
                break
            bins = _simulate_aco_fast(W, capacity, H, seed=seed)
            scores.append(bins)
        
        if len(scores) == 3:
            avg = np.mean(scores)
            if avg < best_score:
                best_score = avg
                best_H = H.copy()
    
    # Fallback if no candidates evaluated
    if best_H is None:
        # Use default: alpha=0.3, gap_power=1.0
        base_term = item_size / safe_gaps
        fill_bonus = 1.0 + 0.3 * (fill_ratio ** 2)
        H_valid = base_term * fill_bonus
        best_H = np.where(fits, H_valid, 1e-9)
        best_H = np.where(np.isfinite(best_H) & (best_H > 0), best_H, 1e-9)
    
    return best_H


def _simulate_aco_fast(demands, capacity, H, ants=20, iters=30, seed=42):
    """Fast ACO simulation to evaluate heuristic quality."""
    rng = np.random.RandomState(seed)
    n = len(demands)
    W = demands
    
    tau = np.ones((n, n))
    min_bins_global = float('inf')
    
    for it in range(iters):
        min_bins_iter = float('inf')
        
        for ant in range(ants):
            used = np.zeros(n, dtype=bool)
            bins_used = 0
            items_left = n
            
            while items_left > 0:
                bins_used += 1
                # Find unused items
                unused = np.where(~used)[0]
                if len(unused) == 0:
                    break
                
                # Random start
                start_idx = unused[rng.randint(len(unused))]
                used[start_idx] = True
                items_left -= 1
                current = start_idx
                current_load = W[current]
                
                # Fill bin
                while True:
                    # Find fitting items
                    fit_mask = ~used & (current_load + W <= capacity)
                    fit_indices = np.where(fit_mask)[0]
                    
                    if len(fit_indices) == 0:
                        break
                    
                    # Compute probabilities
                    probs = tau[current, fit_indices] * H[current, fit_indices]
                    total = np.sum(probs)
                    
                    if total <= 0:
                        next_idx = fit_indices[rng.randint(len(fit_indices))]
                    else:
                        r = rng.random() * total
                        cum = np.cumsum(probs)
                        next_idx = fit_indices[np.searchsorted(cum, r)]
                    
                    used[next_idx] = True
                    items_left -= 1
                    current_load += W[next_idx]
                    current = next_idx
            
            min_bins_iter = min(min_bins_iter, bins_used)
        
        min_bins_global = min(min_bins_global, min_bins_iter)
        
        # Update pheromone
        tau = tau * 0.9 + (1.0 / min_bins_iter)
    
    return min_bins_global
