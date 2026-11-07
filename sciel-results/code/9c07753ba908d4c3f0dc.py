"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack problem.
    
    Args:
        prizes: array of shape (5, 100) with uniform values in [0, 1]
        weights: array of shape (5, 100) with uniform values in [0, 1]
    
    Returns:
        Array of shape (100,) with one attractiveness score per item.
    """
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Validate shapes
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        raise ValueError("Expected prizes and weights to have shape (5, 100)")
    
    # Check for non-finite values
    if not np.all(np.isfinite(prizes)) or not np.all(np.isfinite(weights)):
        raise ValueError("Non-finite values detected in input")
    
    # Capacity per constraint is 25
    capacity = 25.0
    
    # For each item, compute how "tight" it is across all constraints
    # An item that uses a lot of capacity relative to its prize is less attractive
    
    # Compute the total weight vector for each item (already given as columns)
    # weights[:, i] is the weight vector for item i
    
    # Compute the "efficiency" of each item: prize per unit of "bottleneck" constraint
    # The bottleneck constraint for an item is the one where weight/capacity is largest
    
    # For each item, find the maximum ratio of weight to capacity across constraints
    # This represents how much of the tightest constraint this item consumes
    weight_ratios = weights / capacity  # shape (5, 100)
    max_weight_ratio = np.max(weight_ratios, axis=0)  # shape (100,)
    
    # Total prize for each item (sum across constraints? No, prizes are per-constraint)
    # Actually, prizes[i, j] is the prize contribution of item j to constraint i's objective?
    # Re-reading: "prizes and weights, of shape (5, 100), are uniform in [0, 1]"
    # "Every constraint has capacity 25"
    # "Each ant adds items that still fit in all constraints"
    # "highest total prize found by any ant"
    
    # I think prizes[i, j] is the prize value of item j, and there are 5 constraints.
    # But wait, if prizes is (5, 100), maybe each row is a different prize dimension?
    # Or maybe the "total prize" is the sum of prizes across all constraints for the selected items?
    
    # Let me re-read: "highest total prize found by any ant"
    # I think the total prize is the sum of prizes for all items selected by the ant.
    # If prizes is (5, 100), perhaps the total prize for a set of items S is sum_{i in S} sum_{c=0}^{4} prizes[c, i]?
    # Or maybe prizes[c, i] is the prize of item i, and it's the same across constraints? That doesn't make sense.
    
    # Alternative interpretation: There are 5 constraints, and for each constraint c, item i has weight weights[c, i]
    # and prize prizes[c, i]. The total prize is the sum over all constraints and selected items.
    # So total_prize = sum_{c=0}^{4} sum_{i in S} prizes[c, i]
    
    # In that case, the total prize contribution of item i is sum_{c=0}^{4} prizes[c, i]
    
    total_prize_per_item = np.sum(prizes, axis=0)  # shape (100,)
    
    # Now, we want a heuristic that favors items with high total prize and low "cost" in terms of
    # how much they consume the tightest constraint
    
    # A good heuristic: total_prize / (max_weight_ratio + epsilon)
    # This favors items that give a lot of prize relative to how much they strain the tightest constraint
    
    epsilon = 1e-9
    
    # Also consider the sum of weights as a secondary factor
    total_weight_per_item = np.sum(weights, axis=0)  # shape (100,)
    
    # Combine: prize efficiency considering both the bottleneck and total weight
    # heuristic = total_prize / (max_weight_ratio * total_weight + epsilon)
    
    # But let's think about what makes an item attractive in ACO:
    # We want items that give high prize while consuming little capacity.
    # The probability of selecting an item is proportional to pheromone * heuristic.
    # So we want heuristic to be high for "good" items.
    
    # A simple and effective heuristic: prize per unit of total weight
    # But we should also consider the bottleneck constraint.
    
    # Let's use: total_prize / (max_weight_ratio + total_weight/capacity + epsilon)
    # This penalizes items that are heavy in any constraint.
    
    avg_weight_ratio = total_weight_per_item / (5 * capacity)  # average fraction of capacity used
    
    # Heuristic: high prize, low bottleneck, low average weight
    heuristic = total_prize_per_item / (max_weight_ratio + avg_weight_ratio + epsilon)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Normalize to a reasonable range (optional, but can help with numerical stability)
    # The task says "Heuristic entries that are not positive become 1e-9"
    # So we just need to ensure positivity.
    
    return heuristic
# EVOLVE-BLOCK-END
