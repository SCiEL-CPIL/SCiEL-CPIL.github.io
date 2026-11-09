import numpy as np
import time

def _bfd_assignment(demands, capacity):
    n = len(demands)
    sorted_indices = np.argsort(-demands)
    
    bins = [] # List of lists of indices
    bin_loads = [] # List of current loads
    item_bin = np.full(n, -1, dtype=int)
    
    for idx in sorted_indices:
        size = demands[idx]
        best_bin = -1
        best_remaining = float('inf')
        
        for b_idx in range(len(bins)):
            remaining = capacity - bin_loads[b_idx]
            if remaining >= size:
                if remaining < best_remaining:
                    best_remaining = remaining
                    best_bin = b_idx
        
        if best_bin != -1:
            bins[best_bin].append(idx)
            bin_loads[best_bin] += size
            item_bin[idx] = best_bin
        else:
            bins.append([idx])
            bin_loads.append(size)
            item_bin[idx] = len(bins) - 1
            
    return item_bin

def _ffd_assignment(demands, capacity):
    n = len(demands)
    sorted_indices = np.argsort(-demands)
    
    bins = []
    bin_loads = []
    item_bin = np.full(n, -1, dtype=int)
    
    for idx in sorted_indices:
        size = demands[idx]
        placed = False
        for b_idx in range(len(bins)):
            if bin_loads[b_idx] + size <= capacity:
                bins[b_idx].append(idx)
                bin_loads[b_idx] += size
                item_bin[idx] = b_idx
                placed = True
                break
        if not placed:
            bins.append([idx])
            bin_loads.append(size)
            item_bin[idx] = len(bins) - 1
            
    return item_bin

def _fast_aco_eval(demands, capacity, heuristic, n_ants=10, n_iters=10, seed=42):
    n = len(demands)
    rng = np.random.default_rng(seed)
    pheromone = np.ones((n, n))
    best_bins = n
    
    # Precompute fit mask
    pair_sum = demands[:, None] + demands[None, :]
    fits_mask = (pair_sum <= capacity)
    
    # Ensure heuristic is positive
    h = np.maximum(heuristic, 1e-9)
    
    for _ in range(n_iters):
        iteration_best = n
        for _ in range(n_ants):
            placed = np.zeros(n, dtype=bool)
            bins = 0
            current = -1
            
            while not placed.all():
                if current == -1:
                    # Open new bin
                    available = np.where(~placed)[0]
                    if len(available) == 0: break
                    current = available[rng.integers(len(available))]
                    placed[current] = True
                    bins += 1
                else:
                    available = np.where(~placed)[0]
                    if len(available) == 0: break
                    
                    # Filter by fit
                    fit_indices = available[fits_mask[current, available]]
                    if len(fit_indices) == 0:
                        current = -1
                        continue
                    
                    # Calculate probabilities
                    probs = pheromone[current, fit_indices] * h[current, fit_indices]
                    total = probs.sum()
                    if total <= 0:
                        current = fit_indices[rng.integers(len(fit_indices))]
                    else:
                        # Normalize and sample
                        probs_norm = probs / total
                        r = rng.random()
                        cumsum = np.cumsum(probs_norm)
                        idx = np.searchsorted(cumsum, r)
                        if idx >= len(fit_indices):
                            idx = len(fit_indices) - 1
                        current = fit_indices[idx]
                    placed[current] = True
                    
            iteration_best = min(iteration_best, bins)
            
        # Update pheromone
        pheromone *= 0.9
        if iteration_best > 0:
            pheromone += 1.0 / iteration_best
            
        best_bins = min(best_bins, iteration_best)
        
    return best_bins

def heuristics(demands, capacity):
    start = time.time()
    deadline = 280.0 # Safety margin
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # 1. Compute Structural Priors
    item_bin_bfd = _bfd_assignment(demands, capacity)
    item_bin_ffd = _ffd_assignment(demands, capacity)
    
    same_bin_bfd = (item_bin_bfd[:, None] == item_bin_bfd[None, :]).astype(float)
    same_bin_ffd = (item_bin_ffd[:, None] == item_bin_ffd[None, :]).astype(float)
    
    np.fill_diagonal(same_bin_bfd, 0)
    np.fill_diagonal(same_bin_ffd, 0)
    
    # 2. Compute Tightness Prior (Standard Slack)
    pair_sum = demands[:, None] + demands[None, :]
    fits = (pair_sum <= capacity)
    remaining = capacity - pair_sum
    eps = 1e-9
    # High tightness for small remaining space
    tightness = np.where(fits, 1.0 / (remaining + eps), 0.0)
    np.fill_diagonal(tightness, 0)
    
    # 3. Compute Complementarity / Structural Fit Prior
    # Idea: Reward pairs that leave a residual capacity that is a near-integer multiple of the smallest item size.
    # This encourages packing items that "fit together" in terms of granularity, potentially reducing wasted space
    # due to size mismatches, which simple best-fit might miss if a slightly larger item fits but leaves awkward space.
    
    min_item = np.min(demands)
    if min_item > 0:
        # For each pair (i, j), calculate the residual space if they were in a bin together.
        # residual = capacity - (d_i + d_j)
        # We want to reward if residual is close to k * min_item for some integer k >= 0.
        # This is equivalent to minimizing the distance to the nearest multiple of min_item.
        
        # Vectorized calculation
        residuals = capacity - pair_sum
        # Only consider pairs that fit (residuals >= 0)
        # Distance to nearest multiple:
        # multiples = floor(residuals / min_item) * min_item
        # next_multiple = (floor(residuals / min_item) + 1) * min_item
        # dist = min(abs(residuals - multiples), abs(residuals - next_multiple))
        
        # To avoid division by zero or issues with large numbers, we can just use modulo?
        # No, modulo gives distance to 0.
        # Let's use the distance to the nearest multiple.
        
        k = np.floor(residuals / min_item)
        lower_mult = k * min_item
        upper_mult = (k + 1) * min_item
        
        dist_lower = np.abs(residuals - lower_mult)
        dist_upper = np.abs(residuals - upper_mult)
        dist_to_multiple = np.minimum(dist_lower, dist_upper)
        
        # Score: 1 / (dist_to_multiple + eps)
        # If residual is exactly a multiple, dist is 0, score is high.
        # If residual is halfway, score is low.
        
        struct_fit = np.where(fits, 1.0 / (dist_to_multiple + eps), 0.0)
        np.fill_diagonal(struct_fit, 0)
        
        # Normalize struct_fit to be comparable to tightness
        max_struct = np.max(struct_fit)
        if max_struct > 0:
            struct_fit = struct_fit / max_struct
    else:
        struct_fit = np.zeros((n, n))
        
    # 4. Generate Candidates
    candidates = []
    names = []
    
    # C1: BFD Structure * Tightness
    h1 = same_bin_bfd * tightness
    candidates.append(h1); names.append("BFD_Tight")
    
    # C2: FFD Structure * Tightness
    h2 = same_bin_ffd * tightness
    candidates.append(h2); names.append("FFD_Tight")
    
    # C3: (BFD + FFD) * Tightness
    h3 = (same_bin_bfd + same_bin_ffd) * tightness
    candidates.append(h3); names.append("Mix_Tight")
    
    # C4: Pure Tightness
    h4 = tightness.copy()
    candidates.append(h4); names.append("Tight")
    
    # C5: Pure BFD Structure
    h5 = same_bin_bfd.copy()
    candidates.append(h5); names.append("BFD")
    
    # C6: Pure FFD Structure
    h6 = same_bin_ffd.copy()
    candidates.append(h6); names.append("FFD")
    
    # C7: BFD Structure + Tightness (Additive)
    h7 = same_bin_bfd + tightness
    candidates.append(h7); names.append("BFD_Add_Tight")
    
    # C8: Soft BFD (1 if same bin, 0.1 if not) * Tightness
    soft_bfd = np.where(same_bin_bfd > 0, 1.0, 0.1)
    h8 = soft_bfd * tightness
    candidates.append(h8); names.append("Soft_BFD_Tight")
    
    # C9: BFD Structure * (1 + Tightness)
    h9 = same_bin_bfd * (1.0 + tightness)
    candidates.append(h9); names.append("BFD_1_Tight")
    
    # C10: FFD Structure * (1 + Tightness)
    h10 = same_bin_ffd * (1.0 + tightness)
    candidates.append(h10); names.append("FFD_1_Tight")
    
    # New Candidates with Structural Fit
    
    # C11: BFD Structure * (Tightness + Structural Fit)
    h11 = same_bin_bfd * (tightness + struct_fit)
    candidates.append(h11); names.append("BFD_Tight_Struct")
    
    # C12: FFD Structure * (Tightness + Structural Fit)
    h12 = same_bin_ffd * (tightness + struct_fit)
    candidates.append(h12); names.append("FFD_Tight_Struct")
    
    # C13: Pure (Tightness + Structural Fit)
    h13 = tightness + struct_fit
    candidates.append(h13); names.append("Tight_Struct")
    
    # C14: BFD Structure * Tightness * (1 + Structural Fit)
    h14 = same_bin_bfd * tightness * (1.0 + struct_fit)
    candidates.append(h14); names.append("BFD_Tight_Mult_Struct")
    
    # C15: FFD Structure * Tightness * (1 + Structural Fit)
    h15 = same_bin_ffd * tightness * (1.0 + struct_fit)
    candidates.append(h15); names.append("FFD_Tight_Mult_Struct")
    
    # Normalize candidates to max 1.0 for fair initial comparison
    normalized_candidates = []
    for h in candidates:
        h_max = np.max(h)
        if h_max > 0:
            h_norm = h / h_max
        else:
            h_norm = h
        h_norm = np.where(np.isfinite(h_norm), h_norm, 1e-9)
        normalized_candidates.append(h_norm)
        
    best_heuristic = normalized_candidates[0]
    best_score = float('inf')
    
    # Evaluate candidates
    # To save time, we might not evaluate all if time is tight, but 15 candidates * ~1s each is fine.
    for idx, h in enumerate(normalized_candidates):
        if time.time() > deadline:
            break
            
        score = _fast_aco_eval(demands, capacity, h, n_ants=10, n_iters=10, seed=42)
        if score < best_score:
            best_score = score
            best_heuristic = h
            
    # Final sanity check
    if not np.all(np.isfinite(best_heuristic)):
        best_heuristic = np.where(np.isfinite(best_heuristic), best_heuristic, 1e-9)
        
    # Ensure positive
    best_heuristic = np.maximum(best_heuristic, 1e-9)
    
    return best_heuristic
