"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _simulate_aco(W, capacity, H, ants=20, iters=30, seed=42):
    """Fast ACO simulation to evaluate a heuristic matrix."""
    rng = np.random.RandomState(seed)
    n = len(W)
    tau = np.ones((n, n))
    min_bins_global = float('inf')
    
    for it in range(iters):
        bins_counts = []
        for ant in range(ants):
            used = np.zeros(n, dtype=bool)
            bins_used = 0
            items_left = n
            
            while items_left > 0:
                bins_used += 1
                # Random fitting item to start bin
                unused = np.where(~used)[0]
                if len(unused) == 0:
                    break
                start_idx = unused[rng.randint(len(unused))]
                used[start_idx] = True
                items_left -= 1
                current = start_idx
                current_load = W[current]
                
                while True:
                    # Find fitting candidates
                    fits = (~used) & (current_load + W <= capacity)
                    if not np.any(fits):
                        break
                    fit_indices = np.where(fits)[0]
                    probs = tau[current, fit_indices] * H[current, fit_indices]
                    total = probs.sum()
                    if total <= 0:
                        next_idx = fit_indices[rng.randint(len(fit_indices))]
                    else:
                        r = rng.random() * total
                        cum = np.cumsum(probs)
                        next_idx = fit_indices[np.searchsorted(cum, r, side='right')]
                        if next_idx >= len(fit_indices):
                            next_idx = fit_indices[-1]
                    used[next_idx] = True
                    items_left -= 1
                    current_load += W[next_idx]
                    current = next_idx
                    
            bins_counts.append(bins_used)
        
        min_bins_iter = min(bins_counts)
        min_bins_global = min(min_bins_global, min_bins_iter)
        tau = tau * 0.9 + (1.0 / min_bins_iter)
        
    return min_bins_global

def heuristics(demands, capacity):
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s kill
    
    W = np.array(demands, dtype=float)
    n = len(W)
    pair_sums = W[:, None] + W[None, :]
    fit_mask = pair_sums <= capacity
    gaps = capacity - pair_sums
    
    # Precompute base arrays
    eps = 1e-9
    safe_gaps = np.where(fit_mask, gaps + eps, 1.0)
    item_size = W[None, :]  # (1, n) broadcasts to (n, n)
    
    # Build candidate heuristics: (W_j^beta) / (gap^alpha)
    # Include the current program's approach (alpha=1, beta=0) as baseline
    alphas = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    betas = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    candidates = []
    for alpha in alphas:
        for beta in betas:
            if beta == 0:
                numerator = np.ones((n, n))
            else:
                numerator = item_size ** beta
            h = np.where(fit_mask, numerator / (safe_gaps ** alpha), 1e-9)
            h = np.where(h > 0, h, 1e-9)
            if np.all(np.isfinite(h)):
                candidates.append(h)
    
    if not candidates:
        return np.where(fit_mask, 1.0 / safe_gaps, 1e-9)
    
    # Evaluate candidates with a few seeds
    eval_seeds = [42, 123, 999]
    best_H = candidates[0]
    best_score = float('inf')
    
    for H in candidates:
        if time.time() > deadline:
            break
        scores = []
        for seed in eval_seeds:
            if time.time() > deadline:
                break
            bins = _simulate_aco(W, capacity, H, seed=seed)
            scores.append(bins)
        if len(scores) == len(eval_seeds):
            avg = np.mean(scores)
            if avg < best_score:
                best_score = avg
                best_H = H
    
    # Final sanity
    best_H = np.where(np.isfinite(best_H) & (best_H > 0), best_H, 1e-9)
    return best_H
# EVOLVE-BLOCK-END
