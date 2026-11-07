"""Heuristic via EXPERT-SELECTION: evaluate a small set of strong bin-packing
heuristics using a fast deterministic proxy, and select the best one."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _det_greedy_bins(demands, capacity, M, deadline=None):
    """
    Deterministic greedy proxy: simulate the ACO's best-ant behavior by
    greedily picking the item that maximizes M[current, j] among fitting items.
    Starts each new bin with the largest remaining item (FFD-style anchor).
    """
    n = len(demands)
    d = np.asarray(demands, dtype=float)
    total_bins = 0
    
    # Evaluate from multiple starting perspectives: largest, median, smallest
    # This gives a more robust estimate of heuristic quality.
    sorted_idx = np.argsort(d)
    start_indices = [sorted_idx[-1], sorted_idx[n // 2], sorted_idx[0]]
    
    for start_idx in start_indices:
        if deadline is not None and time.time() > deadline:
            # Return current partial result (scaled) to avoid timeout
            if total_bins > 0:
                return total_bins / len(start_indices)
            continue
            
        used = np.zeros(n, dtype=bool)
        bins = 0
        
        while True:
            remaining = np.where(~used)[0]
            if len(remaining) == 0:
                break
            
            # Start new bin with largest remaining item
            start = remaining[np.argmax(d[remaining])]
            used[start] = True
            bin_load = d[start]
            bins += 1
            current = start
            
            while True:
                # Find fitting items
                rem = np.where(~used)[0]
                if len(rem) == 0:
                    break
                fitting = rem[d[rem] <= capacity - bin_load]
                
                if len(fitting) == 0:
                    break
                
                # Pick best j based on M[current, j]
                scores = M[current, fitting]
                best_local = np.argmax(scores)
                best_j = fitting[best_local]
                
                used[best_j] = True
                bin_load += d[best_j]
                current = best_j
                
                if deadline is not None and time.time() > deadline:
                    break
                    
            if deadline is not None and time.time() > deadline:
                break
                
        total_bins += bins
        
    return total_bins / len(start_indices)


def heuristics(demands, capacity):
    demands = np.asarray(demands, dtype=float)
    capacity = float(capacity)
    n = len(demands)
    
    # Deadline guard: allow ~280s out of 330s wall-clock
    t0 = time.time()
    deadline = t0 + 280.0
    
    pair = demands[:, None] + demands[None, :]
    ok = pair <= capacity
    empty = capacity - pair
    ratio = pair / capacity  # 0 to 1 for fitting pairs
    
    candidates = {}
    
    # 1. Tight Fit: 1 / (C - sum + eps). Favors completing the bin.
    M1 = np.full((n, n), 1e-9)
    M1[ok] = 1.0 / (empty[ok] + 1.0)
    candidates['tight'] = M1
    
    # 2. Gaussian: exp(-alpha * (sum - C)^2). Favors near-completion.
    M2 = np.full((n, n), 1e-9)
    M2[ok] = np.exp(-0.1 * (pair[ok] - capacity) ** 2)
    candidates['gauss'] = M2
    
    # 3. Linear Fill: sum. Favors heavy pairs.
    M3 = np.full((n, n), 1e-9)
    M3[ok] = pair[ok]
    candidates['linear'] = M3
    
    # 4. Sharp Tight Fit: 1 / (empty + 0.1)
    M4 = np.full((n, n), 1e-9)
    M4[ok] = 1.0 / (empty[ok] + 0.1)
    candidates['tight_sharp'] = M4
    
    # 5. Hybrid: Tight + Linear
    M5 = np.full((n, n), 1e-9)
    M5[ok] = (1.0 / (empty[ok] + 1.0)) + (pair[ok] / capacity)
    candidates['hybrid'] = M5
    
    # 6. Squared ratio: (pair/capacity)^2 — strongly rewards tight packing
    M6 = np.full((n, n), 1e-9)
    M6[ok] = ratio[ok] ** 2
    candidates['squared_ratio'] = M6
    
    # 7. Cubed ratio: (pair/capacity)^3 — even stronger tightness reward
    M7 = np.full((n, n), 1e-9)
    M7[ok] = ratio[ok] ** 3
    candidates['cubed_ratio'] = M7
    
    # 8. Combined: squared_ratio + tight
    M8 = np.full((n, n), 1e-9)
    M8[ok] = ratio[ok] ** 2 + 1.0 / (empty[ok] + 1.0)
    candidates['sq_plus_tight'] = M8
    
    # 9. Exponential tightness: exp(k * ratio)
    M9 = np.full((n, n), 1e-9)
    M9[ok] = np.exp(2.0 * ratio[ok])
    candidates['exp_ratio'] = M9
    
    # 10. Inverse empty squared: 1/(empty+eps)^2
    M10 = np.full((n, n), 1e-9)
    M10[ok] = 1.0 / ((empty[ok] + 1.0) ** 2)
    candidates['tight_sq'] = M10
    
    # Evaluate each candidate with deadline guard
    best_name = 'tight'
    best_score = float('inf')
    
    for name, M in candidates.items():
        if time.time() > deadline:
            break
        # Safety check
        if not np.all(np.isfinite(M)):
            M[~np.isfinite(M)] = 1e-9
        score = _det_greedy_bins(demands, capacity, M, deadline=deadline)
        if score < best_score:
            best_score = score
            best_name = name
            
    # Return the best matrix
    M_best = candidates[best_name].copy()
    M_best[~np.isfinite(M_best)] = 1e-9
    M_best[M_best <= 0] = 1e-9
    return M_best
# EVOLVE-BLOCK-END
