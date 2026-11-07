"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Computes one attractiveness score per item for the ant colony optimization.
    
    Args:
        prizes: np.array of shape (5, 100) with prizes in [0, 1].
        weights: np.array of shape (5, 100) with weights in [0, 1].
        
    Returns:
        np.array of shape (100,) with heuristic scores.
    """
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Basic validation to prevent exceptions
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        return np.ones(100) * 1e-9
        
    # Check for non-finite values
    if not np.all(np.isfinite(prizes)) or not np.all(np.isfinite(weights)):
        return np.ones(100) * 1e-9

    # Deadline guard (though this function is fast, it's good practice per directive)
    # The main computation is O(N log N) for ranking, which is extremely fast for N=100.
    
    # 1. Calculate total prize and total weight per item
    total_prize = np.sum(prizes, axis=0)
    total_weight = np.sum(weights, axis=0)
    
    # 2. Calculate the "bottleneck constraint" penalty.
    # For each item, look at the ratio of weight to capacity for each constraint.
    # Capacity is 25 for all constraints.
    # weight[i] / 25 is the fraction of capacity consumed by item i in constraint i.
    # If an item consumes a large fraction of a constraint, it's "heavy" for that constraint.
    # We want to penalize items that are "heavy" in any constraint relative to their prize.
    
    # Fraction of capacity used per constraint
    cap_fraction = weights / 25.0
    
    # The maximum fraction of any single constraint used by this item
    max_cap_fraction = np.max(cap_fraction, axis=0)
    
    # 3. Compute a composite efficiency score.
    # Base efficiency: Prize / Total Weight
    # We use a soft minimum or a penalty to avoid division by zero and to penalize high weight.
    # Let's use a rank-based approach to make it scale-invariant.
    
    # Step 3a: Rank items by total prize (higher is better)
    # np.argsort(np.argsort(x)) gives ranks 0..n-1. We want higher ranks for better items.
    prize_rank = np.argsort(np.argsort(total_prize))
    
    # Step 3b: Rank items by total weight (lower is better)
    # Invert the rank: higher rank means lower weight.
    weight_rank = np.argsort(np.argsort(-total_weight))
    
    # Step 3c: Rank items by "bottleneck" weight (lower is better)
    bottleneck_rank = np.argsort(np.argsort(-max_cap_fraction))
    
    # Step 3d: Combine ranks.
    # A simple weighted sum of ranks.
    # We want high prize rank, high weight rank (low weight), high bottleneck rank (low bottleneck).
    # Since all ranks are 0..99, we can sum them.
    # To give more importance to prize, we can weight it.
    # Let's try: 2 * prize_rank + 1 * weight_rank + 1 * bottleneck_rank
    # This makes the heuristic proportional to the combined "goodness" rank.
    
    combined_rank = 2.0 * prize_rank + 1.0 * weight_rank + 1.0 * bottleneck_rank
    
    # Normalize to [0, 1] range? Or just use the rank value?
    # The ACO uses pheromone * heuristic. If heuristic is a rank (0-99), it's a strong signal.
    # If we normalize to [0,1], it's a weaker signal.
    # The seed used ratio, which can vary widely. Ranks are bounded.
    # Let's scale the ranks to be in a reasonable range, e.g., [1, 100] or similar.
    # Or simply use the rank itself. Since pheromone starts at 1, and heuristic is rank,
    # the probability is proportional to 1 * rank.
    
    # To ensure positive values and avoid 0 probability (which would be 1e-9 anyway),
    # we can add 1 to the rank.
    eta = combined_rank + 1.0
    
    # Ensure all values are positive
    eta = np.maximum(eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
