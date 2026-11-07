"""Heuristic via EXPERT-SELECTION: evaluate a small set of strong bin-packing
heuristics using a fast deterministic proxy, and select the best one."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _det_greedy_bins(demands, capacity, M):
    """
    Deterministic greedy proxy:
    Simulate ACO-style greedy selection with multiple starting points.
    """
    n = len(demands)
    d = np.asarray(demands, dtype=float)
    
    # Use top 3 largest items as starting points for robustness
    start_indices = np.argsort(d)[-3:]
    
    total_bins = 0
    for start_idx in start_indices:
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
                fitting = [j for j in rem if d[j] <= capacity - bin_load]
                
                if len(fitting) == 0:
                    break
                
                # Pick best j based on M
                best_j = -1
                best_score = -1
                for j in fitting:
                    score = M[current, j]
                    if score > best_score:
                        best_score = score
                        best_j = j
                
                if best_j == -1:
                    break
                    
                used[best_j] = True
                bin_load += d[best_j]
                current = best_j
                
        total_bins += bins
        
    return total_bins / len(start_indices)


def heuristics(demands, capacity):
    start_time = time.time()
    deadline = start_time + 280  # Leave 50s safety margin for 330s limit
    
    demands = np.asarray(demands, dtype=float)
    capacity = float(capacity)
    n = len(demands)
    
    # Candidate Heuristics
    candidates = {}
    
    # Precompute pair sums and validity mask
    pair = demands[:, None] + demands[None, :]
    ok = pair <= capacity
    empty = capacity - pair
    
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
    
    # 4. Sharp Tight Fit: 1 / (C - sum + 0.1). More aggressive.
    M4 = np.full((n, n), 1e-9)
    M4[ok] = 1.0 / (empty[ok] + 0.1)
    candidates['tight_sharp'] = M4
    
    # 5. Hybrid: Tight + Linear
    M5 = np.full((n, n), 1e-9)
    M5[ok] = (1.0 / (empty[ok] + 1.0)) + (pair[ok] / capacity)
    candidates['hybrid'] = M5
    
    # 6. SQUARED RATIO: (pair/capacity)^2 - strongly rewards tight packing (measured improvement)
    M6 = np.full((n, n), 1e-9)
    M6[ok] = (pair[ok] / capacity) ** 2
    candidates['squared_ratio'] = M6
    
    # 7. SQUARED RATIO + Tight Fit hybrid
    M7 = np.full((n, n), 1e-9)
    M7[ok] = (pair[ok] / capacity) ** 2 + (1.0 / (empty[ok] + 1.0))
    candidates['squared_tight'] = M7
    
    # 8. Cubed ratio for even stronger preference
    M8 = np.full((n, n), 1e-9)
    M8[ok] = (pair[ok] / capacity) ** 3
    candidates['cubed_ratio'] = M8
    
    # Evaluate each candidate with time guard
    best_name = 'tight'
    best_score = float('inf')
    
    for name, M in candidates.items():
        # Check deadline
        if time.time() > deadline:
            break
            
        # Safety check
        if not np.all(np.isfinite(M)):
            M[~np.isfinite(M)] = 1e-9
        
        score = _det_greedy_bins(demands, capacity, M)
        if score < best_score:
            best_score = score
            best_name = name
            
    # Return the best matrix
    M_best = candidates[best_name].copy()
    M_best[~np.isfinite(M_best)] = 1e-9
    M_best[M_best <= 0] = 1e-9
    return M_best
# EVOLVE-BLOCK-END
