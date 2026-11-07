"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Fixed deadline with generous safety margin (evaluator kills at 330s).
_DEADLINE = 330.0
_START_TIME = time.time()


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack problem.
    
    Strategy:
    Combines a scarcity-weighted geometric mean of per-constraint prize-to-weight ratios
    with a soft-max bottleneck penalty.
    
    1. Per-constraint efficiency: prize_c / weight_c for each constraint c and item i.
    2. Scarcity weighting: constraints with higher total demand relative to capacity
       (25) are weighted more heavily, reflecting that they are more likely binding.
    3. Weighted geometric mean: an item is highly rated only if it is efficient across
       all constraints, with tighter constraints contributing more to the score.
    4. Soft-max bottleneck penalty: instead of a hard min (which creates discontinuities),
       we use a log-sum-exp based soft minimum to smoothly penalize items that are
       inefficient in at least one constraint. This is weighted by the top-k/mean 
       scarcity ratio to focus on the most binding constraints.
    
    Args:
        prizes: numpy array of shape (5, 100) with prizes for each constraint/item.
        weights: numpy array of shape (5, 100) with weights for each constraint/item.
        
    Returns:
        numpy array of shape (100,) with heuristic scores (larger is more attractive).
    """
    # Check deadline
    if time.time() - _START_TIME > _DEADLINE - 5.0:
        # Fallback to simple geometric mean if running out of time
        p = np.clip(prizes, 1e-9, None)
        w = np.clip(weights, 1e-9, None)
        ratios = p / w
        log_ratios = np.log(ratios)
        mean_log_ratio = np.mean(log_ratios, axis=0)
        heuristic = np.exp(mean_log_ratio)
        heuristic = np.maximum(heuristic, 1e-9)
        mean_h = np.mean(heuristic)
        if mean_h > 0:
            heuristic = heuristic / mean_h
        return heuristic

    # Ensure inputs are float64 arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    eps = 1e-9
    capacity = 25.0
    
    # 1. Calculate per-constraint ratios: Prize / Weight
    safe_weights = np.clip(weights, eps, None)
    safe_prizes = np.clip(prizes, eps, None)
    
    ratios = safe_prizes / safe_weights  # Shape (5, 100)
    log_ratios = np.log(ratios)  # Shape (5, 100)
    
    # 2. Calculate scarcity for each constraint
    # Total weight of all items in each constraint
    total_weights_per_constraint = np.sum(weights, axis=1)  # Shape (5,)
    
    # Scarcity: how much of the capacity is "demanded"
    # If total_weight > capacity, the constraint is binding
    scarcity = np.clip(total_weights_per_constraint / capacity, 1.0, None)
    
    # Normalize scarcity weights so they sum to 1
    sum_scarcity = np.sum(scarcity)
    if sum_scarcity < eps:
        scarcity = np.ones(5)
        sum_scarcity = 5.0
    scarcity_weights = scarcity / sum_scarcity  # Shape (5,)
    
    # 3. Compute Weighted Geometric Mean
    # log(geo_mean) = sum( w_c * log(ratio_c) ) / sum(w_c)
    # Since scarcity_weights already sum to 1, we just do the weighted sum
    weighted_log_sum = np.sum(scarcity_weights[:, np.newaxis] * log_ratios, axis=0)
    
    # 4. Soft-max bottleneck penalty
    # Instead of min, use a soft minimum via log-sum-exp
    # soft_min(x) = -log(sum(exp(-x)))
    # We weight the penalty by scarcity to focus on binding constraints
    # Use a temperature parameter to control how "soft" the minimum is
    temperature = 0.5  # Lower = closer to hard min
    
    # For the soft min, we want to penalize low ratios
    # So we compute: -soft_min(log_ratios) = log(sum(exp(log_ratios / temperature)))
    # But we want this to be a penalty, so we subtract it from the base score
    
    # Actually, let's think differently. The geometric mean already handles the "all constraints" aspect.
    # The bottleneck penalty should specifically penalize items where one constraint is much worse than others.
    
    # Compute the deviation from the mean for each item
    # For each item, compute the log ratio in each constraint
    # The "bottleneck" is the minimum log ratio
    # We use a soft min to smooth this
    
    # Soft min of log_ratios across constraints (axis 0)
    # soft_min(x) = -T * log(mean(exp(-x/T)))
    # where T is temperature
    
    # For each item, compute soft min of log_ratios
    neg_log_ratios = -log_ratios / temperature  # Shape (5, 100)
    
    # Subtract max for numerical stability
    max_neg = np.max(neg_log_ratios, axis=0, keepdims=True)
    exp_neg = np.exp(neg_log_ratios - max_neg)  # Shape (5, 100)
    
    # Weighted soft min
    # We weight by scarcity, so tight constraints contribute more to the bottleneck
    weighted_exp_neg = scarcity_weights[:, np.newaxis] * exp_neg
    sum_weighted_exp_neg = np.sum(weighted_exp_neg, axis=0)  # Shape (100,)
    
    # soft_min = -temperature * log(sum_weighted_exp_neg) - max_neg * temperature
    # Wait, let's redo this properly.
    
    # Weighted soft min:
    # w_soft_min(x, w) = -T * log(sum_c w_c * exp(-x_c / T))
    
    # Let's compute this correctly
    # exp(-x_c / T) where x_c = log_ratios[c, i]
    # = exp(-log_ratios[c, i] / T)
    
    # For numerical stability, subtract the max of (-x_c/T) across c for each i
    # -x_c/T = -log_ratios[c, i] / T
    
    neg_log_div_T = -log_ratios / temperature  # Shape (5, 100)
    max_neg_log_div_T = np.max(neg_log_div_T, axis=0, keepdims=True)  # Shape (1, 100)
    
    exp_neg_log_div_T = np.exp(neg_log_div_T - max_neg_log_div_T)  # Shape (5, 100)
    
    # Weighted sum
    weighted_exp_sum = np.sum(scarcity_weights[:, np.newaxis] * exp_neg_log_div_T, axis=0)  # Shape (100,)
    
    # Ensure positive
    weighted_exp_sum = np.maximum(weighted_exp_sum, eps)
    
    # Soft min
    w_soft_min = -temperature * (np.log(weighted_exp_sum) + max_neg_log_div_T.squeeze(axis=0))
    
    # The geometric mean score is exp(weighted_log_sum)
    geo_mean_score = np.exp(weighted_log_sum)
    
    # The bottleneck score is w_soft_min (this is the soft minimum of log ratios)
    # We want to penalize items with low bottleneck scores
    # So we combine: heuristic = geo_mean_score * exp(alpha * w_soft_min)
    # where alpha controls the strength of the bottleneck penalty
    
    # Actually, since geo_mean_score = exp(weighted_log_sum) and w_soft_min is on the log scale,
    # we can combine them on the log scale:
    # log_heuristic = weighted_log_sum + alpha * w_soft_min
    
    # But w_soft_min is already the soft min of log_ratios, and weighted_log_sum is the weighted mean.
    # The difference between the mean and the min indicates how "balanced" the item is.
    # Items with high variance across constraints have low w_soft_min relative to weighted_log_sum.
    
    # Let's use: log_heuristic = weighted_log_sum + alpha * (w_soft_min - weighted_log_sum)
    # = (1 - alpha) * weighted_log_sum + alpha * w_soft_min
    
    # This is a convex combination of the weighted mean and the weighted soft min.
    # alpha = 0 gives pure geometric mean, alpha = 1 gives pure soft min.
    
    alpha = 0.3  # Blend factor
    
    log_heuristic = (1 - alpha) * weighted_log_sum + alpha * w_soft_min
    
    # Convert back to linear scale
    heuristic = np.exp(log_heuristic)
    
    # 5. Ensure positivity and finiteness
    heuristic = np.nan_to_num(heuristic, nan=1e-9, posinf=1e9, neginf=1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    # 6. Normalize to have mean 1
    # This stabilizes the interaction with pheromones which start at 1.0.
    mean_h = np.mean(heuristic)
    if mean_h > 0:
        heuristic = heuristic / mean_h
        
    return heuristic
# EVOLVE-BLOCK-END
