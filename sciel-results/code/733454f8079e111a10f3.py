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
    
    # Calculate a "soft-min" penalty: items that are weak (low weight) in any constraint
    # are actually GOOD (they consume less), so we want to reward low weights.
    # But we want to penalize items that are heavy in ANY constraint.
    # A common trick is to use the sum, but we can also look at the distribution.
    
    # Strategy:
    # 1. Base efficiency: prize / total_weight
    # 2. Bottleneck penalty: items with high max_weight are risky.
    # 3. Balance bonus: items with low variance in weights are "balanced".
    
    # Let's try a composite cost:
    # cost = alpha * weight_sums + beta * weight_maxs * n_constraints
    # If alpha=1, beta=0, it's the baseline.
    # If alpha=0, beta=1, it's max * 5 (roughly comparable scale to sum).
    
    # From previous observations, sum alone scored 1.0.
    # The inspiration suggests soft-min/geometric mean helps.
    # Let's try: cost = weight_sums + gamma * (weight_maxs - weight_sums/n_constraints)
    # This adds a penalty for imbalance (items that are heavy in one dim but light in others).
    # If an item is balanced, max == sum/n, penalty is 0.
    # If an item is unbalanced, max > sum/n, penalty is positive.
    
    imbalance = weight_maxs - weight_sums / n_constraints
    cost = weight_sums + 0.5 * imbalance + 1e-9
    
    heuristic = prize_vector / cost
    
    # Amplify the heuristic to focus on high-value items
    # This makes the probability distribution steeper
    # Previous run used 1.1, let's try a slightly stronger amplification
    heuristic = heuristic ** 1.15
    
    # Ensure positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Check time
    if time.time() > deadline:
        return heuristic

    return heuristic

# EVOLVE-BLOCK-END
