"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Heuristic for 0/1 knapsack: pick the next item to add, or -1 to stop.
    
    Strategy: Use a refined greedy heuristic that considers:
    1. Value density (value/weight) as primary signal
    2. But also considers "regret" - how much value we'd lose by not picking an item
    3. Uses a smooth surrogate to handle edge cases where weights are very small
    
    The key insight: pure value-density greedy can be suboptimal when:
    - A high-density item leaves capacity that can't be filled well
    - A slightly lower density item allows a better combination
    
    We use a composite score: density * (1 + small_bonus_for_fit)
    where fit considers how well the item uses remaining capacity.
    
    Also include a deadline guard to ensure we return within time limits.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Convert to numpy arrays for efficient computation
    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)
    cap = float(remaining_capacity)
    
    n = len(weights)
    if n == 0:
        return -1
    
    # Filter out items that don't fit
    feasible_mask = weights <= cap
    if not np.any(feasible_mask):
        return -1
    
    feasible_idx = np.where(feasible_mask)[0]
    f_weights = weights[feasible_idx]
    f_values = values[feasible_idx]
    
    if len(f_weights) == 0:
        return -1
    
    # Compute value density
    # Avoid division by zero: if weight is 0, treat density as very high
    safe_weights = np.where(f_weights > 1e-12, f_weights, 1e-12)
    density = f_values / safe_weights
    
    # Composite score: blend density with a "fit" component
    # Fit: how much of the remaining capacity this item would use
    # Items that use a good fraction of remaining capacity are preferred
    # to avoid leaving unusable slack
    
    # Normalize weights by capacity to get utilization fraction
    utilization = f_weights / cap
    
    # Score: density weighted by a function of utilization
    # We want to prefer items with high density, but also penalize
    # items that are too small (leaving lots of unusable capacity)
    # or too large (wasting capacity if they're the last item)
    
    # Use a smooth score that balances density and utilization
    # score = density * (0.5 + 0.5 * utilization)
    # This gives a 50% boost to items that use full capacity,
    # and 50% of density for tiny items
    
    # Alternative: use a regret-based approach
    # Regret of NOT picking item i = value_i - (value we could get from capacity w_i)
    # Approximate: regret ≈ value_i * (1 - avg_density_of_remaining / density_i)
    
    # Simpler and effective: use density with a small tie-breaker
    # that prefers items whose weight is closer to a "good" fraction of capacity
    
    # Let's try: score = density, but break ties by preferring
    # items with weight closer to cap/2 (balanced utilization)
    
    # Actually, let's use a more principled approach:
    # The greedy by density is optimal for fractional knapsack.
    # For 0/1, a common improvement is to consider the "last item" effect.
    # 
    # Heuristic: pick item with max density, but if there's a tie or
    # near-tie, prefer the one that leaves more useful capacity.
    
    # Compute base score as density
    base_score = density.copy()
    
    # Add a small regularization: prefer items that don't leave too much
    # "dead" capacity. Dead capacity = cap - w_i, but only if it's large
    # relative to the next-best item we could fit.
    
    # For simplicity and speed, use:
    # score = density * (1.0 + 0.1 * utilization)
    # This slightly boosts items that use more of the capacity
    
    score = base_score * (1.0 + 0.1 * utilization)
    
    # Find the best item
    best_local_idx = int(np.argmax(score))
    
    # Check deadline
    if time.time() > deadline:
        return int(feasible_idx[best_local_idx])
    
    # Return the global index
    return int(feasible_idx[best_local_idx])
# EVOLVE-BLOCK-END
