# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    """
    Returns an n x n matrix of attractiveness for placing item j right after item i.
    
    Strategy: Complementarity graph with structural packing patterns.
    - Reward pairs that leave residual capacity close to residuals seen in a Best-Fit-Decreasing (BFD) solution.
    - Combine with slack/tightness weighting and best-fit bin clustering.
    - Pairs that don't fit get 1e-9
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    n = len(demands)
    demands = np.asarray(demands, dtype=np.float64)
    
    # Best-Fit-Decreasing bin assignment for structural clustering
    order = np.argsort(-demands)
    
    bins = []  # list of lists of original indices
    bin_residuals = []  # remaining capacity per bin
    
    # Track "good" residuals encountered during BFD
    # We consider the residual after each item placement as a "state"
    # And the final residual of each bin as a "terminal state"
    good_residuals = set()
    
    for idx in order:
        # Best-fit: find bin with smallest residual that still fits
        best_b = -1
        best_residual = float('inf')
        for b in range(len(bins)):
            if bin_residuals[b] >= demands[idx]:
                if bin_residuals[b] < best_residual:
                    best_residual = bin_residuals[b]
                    best_b = b
        
        if best_b != -1:
            # Record the residual before placing (which was the bin's residual)
            # And the residual after placing
            good_residuals.add(bin_residuals[best_b])
            bins[best_b].append(idx)
            bin_residuals[best_b] -= demands[idx]
            good_residuals.add(bin_residuals[best_b])
        else:
            bins.append([idx])
            bin_residuals.append(capacity - demands[idx])
            good_residuals.add(capacity - demands[idx])
    
    # Also add 0 as a good residual (perfect fit)
    good_residuals.add(0.0)
    
    # Convert to array for vectorized distance calculation
    good_res_arr = np.array(list(good_residuals), dtype=np.float64)
    
    # Build bin membership: bin_id for each item
    bin_of = np.full(n, -1, dtype=np.int64)
    for b, members in enumerate(bins):
        for idx in members:
            bin_of[idx] = b
    
    # Build heuristic matrix
    pair_sum = demands[:, None] + demands[None, :]
    fits = pair_sum <= capacity
    
    # Residual after placing pair
    residual = capacity - pair_sum
    
    # Structural score: Distance to nearest "good" residual
    # We want to reward residuals that are close to any of the good_residuals
    # dist[i, j] = min_k |residual[i, j] - good_res_arr[k]|
    
    # To avoid O(n^2 * K) which might be slow if K is large, 
    # but n=120, so n^2=14400. K is at most n+1=121.
    # 14400 * 121 = ~1.7M operations. This is fast.
    
    # Reshape residual for broadcasting: (n, n, 1)
    res_reshaped = residual[:, :, None]
    good_reshaped = good_res_arr[None, None, :]
    
    distances = np.abs(res_reshaped - good_reshaped)
    min_dist = np.min(distances, axis=2)
    
    # Scale for exponential decay
    # Use a fraction of the capacity or min_size as sigma
    # min_size is a good scale
    min_size = np.min(demands)
    sigma = 0.5 * min_size
    
    struct_score = np.exp(-min_dist / sigma)
    
    # Tightness score: prefer pairs that leave small residual (tighter pack)
    # Use inverse with offset to avoid division by zero
    tight_score = 1.0 / (1.0 + residual)
    
    # Same-bin bonus: items in the same BFD bin get a multiplier
    same_bin = (bin_of[:, None] == bin_of[None, :])
    same_bin_bonus = np.where(same_bin, 2.0, 1.0) # Slightly reduced bonus to allow cross-bin exploration
    
    # Combine scores
    # Weight structural pattern more heavily
    H = np.where(fits, (2.0 * struct_score + 1.0 * tight_score) * same_bin_bonus, 1e-9)
    
    # Ensure diagonal is not used (i != j), set to 1e-9
    np.fill_diagonal(H, 1e-9)
    
    # Ensure all values are positive and finite
    H = np.maximum(H, 1e-9)
    H = np.where(np.isfinite(H), H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
