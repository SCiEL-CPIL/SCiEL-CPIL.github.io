"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    start_time = time.time()
    deadline = 300.0 - 5.0  # Safety margin

    # Convert to numpy arrays for consistent behavior
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # Handle potential 2D prizes by reducing to 1D if necessary
    if prizes.ndim == 2:
        prize_vector = np.sum(prizes, axis=0)
    else:
        prize_vector = prizes.flatten()

    # Ensure weights is (5, 100)
    n_constraints, n_items = weights.shape

    # Calculate weights sum per item (axis=0 sums over constraints)
    weight_sums = np.sum(weights, axis=0)
    
    # Calculate max weight per item (bottleneck)
    weight_maxs = np.max(weights, axis=0)
    
    # Calculate geometric mean per item
    # Add small epsilon to avoid log(0)
    log_weights = np.log(weights + 1e-9)
    geo_mean = np.exp(np.mean(log_weights, axis=0))
    
    # Strategy based on "bottleneck slack" score:
    # Calculate minimum remaining capacity ratio across all constraints if item were added first.
    # capacity = 25. 
    # slack_c = (25 - w_ic) / w_ic. This is high when w_ic is small.
    # min_slack = min_c slack_c.
    # This directly prioritizes items that fit easily in the tightest constraints.
    
    capacity = 25.0
    
    # Avoid division by zero
    safe_weights = np.maximum(weights, 1e-9)
    
    # Compute slack for each constraint and item
    # slack[c, i] = (capacity - weights[c, i]) / safe_weights[c, i]
    slack = (capacity - weights) / safe_weights
    
    # Find the minimum slack across constraints for each item
    min_slack = np.min(slack, axis=0)
    
    # The heuristic should combine prize and feasibility.
    # The directive suggests: "multiply this feasibility margin by the item's total prize"
    # However, min_slack can be very large for light items, which might dominate.
    # Let's look at the distribution.
    # If w is small, slack is large. If w is close to 25, slack is small.
    # If w > 25, slack is negative.
    
    # We want to reward items that are feasible and have high prize.
    # A simple product: prize * min_slack might work, but scale matters.
    # Alternatively, we can use a ratio or a soft combination.
    
    # Let's try: heuristic = prize * (1 + min_slack)
    # Or perhaps: heuristic = prize * min_slack / (1 + min_slack) to bound it?
    # Or simply: heuristic = prize / (1 / min_slack) = prize * min_slack?
    
    # Let's stick to the directive's suggestion: "multiply this feasibility margin by the item's total prize"
    # But we need to handle negative slacks (infeasible items).
    # If min_slack < 0, the item is infeasible in at least one constraint initially?
    # No, capacity is 25, weights are in [0,1]. So weights are always <= 1.
    # Thus, 25 - w >= 24. slack >= 24.
    # So min_slack is always positive and large (~24 to 25).
    # This means min_slack doesn't vary much (24 to 25).
    # The variation is in the denominator w.
    # So min_slack is essentially proportional to 1/max_weight.
    
    # Let's re-evaluate.
    # slack_c = (25 - w) / w = 25/w - 1.
    # min_slack = min_c (25/w_c - 1) = 25/max_c(w_c) - 1.
    # So min_slack is determined by the maximum weight in any constraint.
    # This is similar to using 1/max_weight.
    
    # The current program uses: cost = sum_w + 0.5 * (max_w - sum_w/5).
    # This is roughly sum_w + 0.5*max_w - 0.1*sum_w = 0.9*sum_w + 0.5*max_w.
    
    # The directive suggests using min_slack directly.
    # Let's try: heuristic = prize * min_slack
    # Since min_slack ~ 25/max_w, this is like prize * 25 / max_w.
    # This puts more weight on the bottleneck constraint than the sum.
    
    # Let's compute heuristic = prize * min_slack
    heuristic = prize_vector * min_slack
    
    # We might need to normalize or adjust scale.
    # Let's compare magnitudes.
    # prize ~ O(1) (sum of 5 uniforms ~ 2.5).
    # min_slack ~ 24-25.
    # heuristic ~ 60.
    
    # The previous heuristic was prize / cost.
    # cost ~ 0.9*2.5 + 0.5*0.5 ~ 2.5.
    # heuristic ~ 1.
    
    # The absolute scale doesn't matter as much as the relative order, 
    # but the ACO probability is proportional to pheromone * heuristic.
    # Pheromone starts at 1.
    # So initial probabilities are proportional to heuristic.
    
    # Let's try to keep the scale similar to avoid extreme probabilities.
    # We can divide by the mean or max.
    # Or just use the ratio.
    
    # Let's try: heuristic = prize / (1.0 / min_slack) = prize * min_slack
    # But to be safe, let's normalize min_slack to be around 1.
    # min_slack is ~25.
    # normalized_slack = min_slack / 25.0
    
    normalized_slack = min_slack / 25.0
    
    # heuristic = prize * normalized_slack
    heuristic = prize_vector * normalized_slack
    
    # Let's also add a small power to sharpen the distribution if needed.
    # But first, let's see if this simple change works.
    # The directive says "multiply this feasibility margin by the item's total prize".
    
    # Ensure positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Check time
    if time.time() > deadline:
        return heuristic

    return heuristic

# EVOLVE-BLOCK-END
