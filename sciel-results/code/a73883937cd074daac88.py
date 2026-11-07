import numpy as np
import time

def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ACO on multiple-constraint knapsack.
    
    Combines:
    1. Scarcity-weighted marginal utility (from inspiration 1)
    2. Soft-max bottleneck penalty via log-sum-exp (from knowledge)
    3. Smooth surrogate for the min objective using weighted log-sum-exp
    
    Strategy:
    - Compute per-constraint efficiency: P[i] / W[c,i]
    - Estimate scarcity per constraint using top-k efficiency ratio
    - Use scarcity to weight the log-sum-exp soft-min of per-constraint ratios
    - This rewards items that are efficient in constraints that are binding
    """
    start_time = time.time()
    deadline = 320.0  # Safety margin before 330s kill
    
    eps = 1e-9
    n_items = 100
    n_constraints = 5
    capacity = 25.0
    
    # Convert to numpy arrays for safety
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Per-constraint efficiency: eff[c, i] = P[i] / W[c, i]
    with np.errstate(divide='ignore', invalid='ignore'):
        eff = prizes[None, :] / (weights + eps)
    eff = np.where(np.isfinite(eff), eff, 0.0)
    
    # Estimate scarcity per constraint using top-k efficiency
    # k = top 20% of items
    k = max(1, n_items // 5)
    top_k_eff = np.zeros(n_constraints)
    for c in range(n_constraints):
        # Use partition for efficiency
        top_k_eff[c] = np.partition(eff[c], -k)[-k].mean()
    
    # Mean efficiency per constraint
    mean_eff = eff.mean(axis=1)
    
    # Scarcity ratio: top_k / mean
    # Higher ratio means the top items are much more efficient than average,
    # suggesting this constraint is more "selective" or binding
    with np.errstate(divide='ignore', invalid='ignore'):
        scarcity_ratio = top_k_eff / (mean_eff + eps)
    scarcity_ratio = np.where(np.isfinite(scarcity_ratio), scarcity_ratio, 1.0)
    
    # Normalize scarcity weights to sum to 1
    scarcity = scarcity_ratio / (scarcity_ratio.sum() + eps)
    
    # Soft-min via log-sum-exp:
    # We want to penalize items that are inefficient in ANY constraint
    # min_c(eff[c,i]) can be approximated by soft-min
    # soft_min(x) = -logsumexp(-x, axis)
    # But we want to weight by scarcity
    
    # Weighted log-sum-exp for soft-min:
    # For each item i, compute: -logsumexp(-scarcity[c] * eff[c,i] / scarcity[c])
    # Actually, let's think differently.
    
    # We want a smooth surrogate for min_c(eff[c,i]) weighted by scarcity
    # One approach: use the weighted geometric mean as a smooth min
    # log(geo) = sum_c(scarcity[c] * log(eff[c,i]))
    # This is a smooth interpolation between arithmetic and geometric means
    
    # But we can do better with log-sum-exp soft-min:
    # soft_min(x_1, ..., x_n) = -logsumexp(-x_1, ..., -x_n)
    # This is smooth and approximates min as the temperature goes to 0
    
    # Let's use a temperature parameter to control smoothness
    # For now, use the standard log-sum-exp without temperature scaling
    # since our values are O(1)
    
    # Compute weighted soft-min:
    # For each item i: soft_min_c(scarcity[c] * eff[c,i])
    # Using log-sum-exp: -logsumexp(-scarcity[:, None] * eff, axis=0)
    
    neg_weighted_eff = -scarcity[:, None] * eff  # (5, 100)
    
    # Log-sum-exp for numerical stability
    max_neg = neg_weighted_eff.max(axis=0)  # (100,)
    log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_weighted_eff - max_neg[:, None]), axis=0))
    
    soft_min = -log_sum_exp  # (100,)
    
    # Also compute the aggregate efficiency as a baseline
    total_prizes = prizes.sum(axis=0)
    total_weights = weights.sum(axis=0)
    H_agg = total_prizes / (total_weights + eps)
    
    # Combine soft-min (bottleneck) with aggregate efficiency
    # Normalize both to mean 1
    H_agg_norm = H_agg / (H_agg.mean() + eps)
    H_agg_norm = np.maximum(H_agg_norm, eps)
    
    soft_min_norm = soft_min / (soft_min.mean() + eps)
    soft_min_norm = np.maximum(soft_min_norm, eps)
    
    # Blend: use geometric mean of the two normalized scores
    # This rewards items that are good on BOTH aggregate and bottleneck
    H = np.sqrt(H_agg_norm * soft_min_norm)
    
    # Ensure positivity and finiteness
    H = np.where(np.isfinite(H), H, eps)
    H = np.maximum(H, eps)
    
    # Deadline guard
    if time.time() > deadline - 5.0:
        return H
    
    return H
