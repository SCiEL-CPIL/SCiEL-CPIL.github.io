"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Compute one attractiveness score per item for the ant colony optimization.
    
    Args:
        prizes: array of shape (5, 100), uniform in [0, 1]
        weights: array of shape (5, 100), uniform in [0, 1]
    
    Returns:
        array of shape (100,) with one score per item (larger = more attractive)
    """
    t0 = time.time()
    DEADLINE = 325.0  # safety margin before 330s kill
    
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Ensure we have the right shape
    if prizes.ndim == 2 and prizes.shape[0] == 5 and prizes.shape[1] == 100:
        p = prizes
    elif prizes.ndim == 1 and prizes.shape[0] == 100:
        p = prizes.reshape(1, 100)
    else:
        flat = prizes.reshape(-1)
        if flat.shape[0] != 100:
            tmp = np.zeros(100)
            n = min(100, flat.shape[0])
            tmp[:n] = flat[:n]
            flat = tmp
        p = flat.reshape(1, 100)
    
    if weights.ndim == 2 and weights.shape[0] == 5 and weights.shape[1] == 100:
        w = weights
    elif weights.ndim == 1 and weights.shape[0] == 100:
        w = weights.reshape(1, 100)
    else:
        flat = weights.reshape(-1)
        if flat.shape[0] != 100:
            tmp = np.zeros(100)
            n = min(100, flat.shape[0])
            tmp[:n] = flat[:n]
            flat = tmp
        w = flat.reshape(1, 100)
    
    n_constraints = p.shape[0]
    n_items = 100
    
    # Total prize per item
    total_prizes = p.sum(axis=0)
    
    # Total weight per item
    total_weights = w.sum(axis=0)
    
    # Per-constraint efficiency: prize_c / weight_c for each constraint
    eff = p / (w + 1e-9)
    
    # Soft-min across constraints with log-sum-exp for numerical stability
    T = 0.4
    log_eff = np.log(eff + 1e-15)
    neg_log_eff_div_T = -log_eff / T
    max_neg = neg_log_eff_div_T.max(axis=0, keepdims=True)
    log_sum_exp = max_neg.squeeze(0) + np.log(np.exp(neg_log_eff_div_T - max_neg).sum(axis=0))
    soft_min = -T * (log_sum_exp - np.log(n_constraints))
    
    # Global ratio
    ratio = total_prizes / (total_weights + 1e-9)
    
    # --- Dynamic context-aware density via greedy packing simulation ---
    # For each item, simulate greedy packing of all OTHER items sorted by 
    # a simple density heuristic, then compute how much residual capacity
    # remains. The marginal value of item i is its prize relative to the
    # tightest residual constraint it would consume.
    
    # Use a quick density ordering for the greedy simulation
    density_order = np.argsort(-ratio)  # highest density first
    
    # Precompute: for each item i, what is the residual capacity after
    # packing all items except i greedily?
    # This is O(n^2 * n_constraints) = 100*100*5 = 50000 ops, very fast.
    
    capacity = 25.0  # fixed per constraint
    
    # Greedy packing residual for each exclusion
    # residual[i][c] = remaining capacity in constraint c after packing all items except i
    residual = np.zeros((n_items, n_constraints))
    
    for i in range(n_items):
        used = np.zeros(n_constraints)
        for j in density_order:
            if j == i:
                continue
            # Check if item j fits
            fits = True
            for c in range(n_constraints):
                if used[c] + w[c, j] > capacity:
                    fits = False
                    break
            if fits:
                used += w[:, j]
        residual[i] = capacity - used
    
    # Marginal value: prize of item i relative to the tightest constraint it would consume
    # For item i, the binding constraint is the one where w[i,c] / residual[i,c] is largest
    # (i.e., the constraint where adding item i is most constraining)
    # If residual is 0 in some constraint, the item can't be added -> low marginal value
    
    # Compute the "bottleneck ratio" for each item:
    # bottleneck[i] = max_c ( w[i,c] / (residual[i,c] + eps) )
    # Lower bottleneck = more attractive (item fits easily)
    eps = 1e-9
    bottleneck_ratio = w / (residual + eps)  # shape (n_constraints, n_items)
    max_bottleneck = bottleneck_ratio.max(axis=0)  # shape (n_items,)
    
    # Marginal density: prize / (total weight * max_bottleneck)
    # Items with high prize, low weight, and low bottleneck score high
    marginal_density = total_prizes / (total_weights * (max_bottleneck + eps) + eps)
    
    # Combine all components:
    # 1. Global ratio (static density)
    # 2. Soft-min efficiency (per-constraint balance)
    # 3. Marginal density (dynamic, context-aware)
    
    # Normalize each component to [0, 1] range for blending
    def normalize(x):
        x = np.asarray(x, dtype=float)
        mn = x.min()
        mx = x.max()
        if mx - mn < 1e-15:
            return np.ones_like(x) * 0.5
        return (x - mn) / (mx - mn)
    
    norm_ratio = normalize(ratio)
    norm_soft_min = normalize(soft_min)
    norm_marginal = normalize(marginal_density)
    
    # Weighted geometric mean with emphasis on marginal density
    # alpha=0.3, beta=0.3, gamma=0.4
    alpha, beta, gamma = 0.3, 0.3, 0.4
    scores = (norm_ratio ** alpha) * (norm_soft_min ** beta) * (norm_marginal ** gamma)
    
    # Sharpen the distribution to make high-scoring items stand out more
    scores = scores ** 1.5
    
    # Ensure all scores are positive
    scores = np.where(scores > 0, scores, 1e-9)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    
    # Deadline check - if we're running long, return what we have
    if time.time() - t0 > DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
