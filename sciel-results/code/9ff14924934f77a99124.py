"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    """
    Returns an n x n matrix of attractiveness for placing item j right after item i.
    
    Strategy: Complementarity graph with structural packing patterns.
    - Reward pairs that leave residual capacity as near-integer multiples of min item size
    - Combine with slack/tightness weighting and best-fit bin clustering
    - Pairs that don't fit get 1e-9
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    n = len(demands)
    demands = np.asarray(demands, dtype=np.float64)
    
    # Find minimum item size for structural pattern detection
    min_size = np.min(demands)
    
    # Multi-strategy bin assignment for structural clustering
    def _bfd(demands, capacity):
        order = np.argsort(-demands)
        bins = []
        bin_residuals = []
        for idx in order:
            best_b = -1
            best_residual = float('inf')
            for b in range(len(bins)):
                if bin_residuals[b] >= demands[idx]:
                    if bin_residuals[b] < best_residual:
                        best_residual = bin_residuals[b]
                        best_b = b
            if best_b != -1:
                bins[best_b].append(idx)
                bin_residuals[best_b] -= demands[idx]
            else:
                bins.append([idx])
                bin_residuals.append(capacity - demands[idx])
        return bins
    
    def _ffd(demands, capacity):
        order = np.argsort(-demands)
        bins = []
        bin_residuals = []
        for idx in order:
            placed = False
            for b in range(len(bins)):
                if bin_residuals[b] >= demands[idx]:
                    bins[b].append(idx)
                    bin_residuals[b] -= demands[idx]
                    placed = True
                    break
            if not placed:
                bins.append([idx])
                bin_residuals.append(capacity - demands[idx])
        return bins
    
    def _wfd(demands, capacity):
        # Worst-fit decreasing: place in bin with largest residual that fits
        order = np.argsort(-demands)
        bins = []
        bin_residuals = []
        for idx in order:
            best_b = -1
            best_residual = -1
            for b in range(len(bins)):
                if bin_residuals[b] >= demands[idx]:
                    if bin_residuals[b] > best_residual:
                        best_residual = bin_residuals[b]
                        best_b = b
            if best_b != -1:
                bins[best_b].append(idx)
                bin_residuals[best_b] -= demands[idx]
            else:
                bins.append([idx])
                bin_residuals.append(capacity - demands[idx])
        return bins
    
    # Run all three strategies and count co-occurrences
    bins_bfd = _bfd(demands, capacity)
    bins_ffd = _ffd(demands, capacity)
    bins_wfd = _wfd(demands, capacity)
    
    # Count how many strategies place each pair in the same bin
    co_count = np.zeros((n, n), dtype=np.float64)
    for bins in [bins_bfd, bins_ffd, bins_wfd]:
        bin_of = np.full(n, -1, dtype=np.int64)
        for b, members in enumerate(bins):
            for idx in members:
                bin_of[idx] = b
        same = (bin_of[:, None] == bin_of[None, :]).astype(np.float64)
        co_count += same
    
    # Build heuristic matrix
    pair_sum = demands[:, None] + demands[None, :]
    fits = pair_sum <= capacity
    
    # Residual after placing pair
    residual = capacity - pair_sum
    
    # Structural pattern: reward residuals that are near-integer multiples of min_size
    # Use smooth soft-min via log-sum-exp for better gradient-like behavior
    if min_size > 0:
        # Distance to nearest integer multiple of min_size
        multiples = np.round(residual / min_size)
        nearest_mult = multiples * min_size
        struct_dist = np.abs(residual - nearest_mult)
        # Smooth exponential decay with tuned temperature
        temp = 0.35 * min_size
        struct_score = np.exp(-struct_dist / temp)
    else:
        struct_score = np.ones((n, n))
    
    # Tightness score: prefer pairs that leave small residual (tighter pack)
    # Use inverse with offset to avoid division by zero
    tight_score = 1.0 / (1.0 + residual)
    
    # Complementarity score: reward pairs where residual is close to some common item size
    # This captures "this pair leaves space that exactly fits another item"
    # Use log-sum-exp soft-min over distances to all item sizes
    if n > 0:
        # For each pair (i,j), compute min distance from residual[i,j] to any demand[k]
        # residual shape: (n, n), demands shape: (n,)
        # dist shape: (n, n, n) -> too large for n=120, use vectorized approach
        # Instead: for each pair, find the demand value closest to the residual
        # Use broadcasting carefully
        # dist[i,j,k] = |residual[i,j] - demands[k]|
        # We want softmin_k dist[i,j,k]
        # For memory efficiency with n=120, this is 120^3 = 1.7M floats ~ 14MB, OK
        dist_to_items = np.abs(residual[:, :, None] - demands[None, None, :])
        # Soft-min via log-sum-exp: softmin(x) = -log(sum(exp(-x/T)))
        # Scale to avoid overflow
        T_comp = 3.0
        neg_scaled = -dist_to_items / T_comp
        max_neg = np.max(neg_scaled, axis=2, keepdims=True)
        exp_sum = np.sum(np.exp(neg_scaled - max_neg), axis=2)
        softmin_dist = -T_comp * np.log(exp_sum + 1e-30) - max_neg[:, :, 0]
        # Complementarity score: lower softmin distance -> higher score
        comp_score = np.exp(-softmin_dist / T_comp)
    else:
        comp_score = np.ones((n, n))
    
    # Multi-strategy bin clustering bonus
    # co_count[i,j] is 0, 1, 2, or 3 depending on how many strategies agree
    # Use smooth bonus: 1 + alpha * co_count
    same_bin_bonus = 1.0 + 0.8 * co_count
    
    # Combine scores: weighted combination of structural, tightness, complementarity, and bin clustering
    # Weight structural and complementarity heavily as they capture packing structure
    H = np.where(fits, (1.2 * struct_score + 0.8 * tight_score + 1.0 * comp_score) * same_bin_bonus, 1e-9)
    
    # Ensure diagonal is not used (i != j), set to 1e-9
    np.fill_diagonal(H, 1e-9)
    
    # Ensure all values are positive and finite
    H = np.maximum(H, 1e-9)
    H = np.where(np.isfinite(H), H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
