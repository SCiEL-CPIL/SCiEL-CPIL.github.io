import numpy as np
import time

def heuristics(prizes, weights):
    """
    Computes attractiveness scores for items in a multiple-constraint knapsack problem.
    
    Args:
        prizes: numpy array of shape (5, 100), uniform in [0, 1].
        weights: numpy array of shape (5, 100), uniform in [0, 1].
        
    Returns:
        numpy array of shape (100,) with one score per item.
    """
    # Ensure inputs are numpy arrays of float64 for precision
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Calculate total prize for each item (sum across constraints)
    # Shape: (100,)
    total_prizes = np.sum(prizes, axis=0)
    
    # Calculate the bottleneck slack score
    # We want to penalize items that are heavy in any constraint.
    # The "slack" after adding item j to constraint i is 25 - w_ij.
    # The ratio (25 - w_ij) / w_ij measures how much extra capacity is left relative to the item's weight.
    # A higher ratio is better (lighter item).
    # We take the minimum across constraints to identify the "bottleneck" constraint for that item.
    
    # Avoid division by zero. Since weights are in [0, 1], they are small but > 0 mostly.
    # If w is 0, the constraint is not binding. We can treat it as very large slack.
    # Using epsilon to prevent inf/nan.
    eps = 1e-9
    safe_weights = np.maximum(weights, eps)
    
    # Calculate slack ratios for each constraint and item
    # Shape: (5, 100)
    slack_ratios = (25.0 - weights) / safe_weights
    
    # If an item has weight 0 in a constraint, the ratio is huge.
    # If an item has weight > 25 (not possible here), ratio is negative.
    
    # Find the minimum slack ratio across constraints for each item
    # This represents the tightest constraint this item would face.
    # Shape: (100,)
    bottleneck_slack = np.min(slack_ratios, axis=0)
    
    # Combine with total prize
    # Heuristic = Prize * Bottleneck_Slack
    heuristic_scores = total_prizes * bottleneck_slack
    
    # Ensure all scores are positive and finite
    # The task states: "Heuristic entries that are not positive become 1e-9"
    heuristic_scores = np.where(heuristic_scores > 0, heuristic_scores, 1e-9)
    
    # Replace any non-finite values (inf, nan) with 1e-9
    heuristic_scores = np.where(np.isfinite(heuristic_scores), heuristic_scores, 1e-9)
    
    return heuristic_scores
