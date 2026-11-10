"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # prizes shape (5, 100), weights shape (5, 100)
    # Each constraint has capacity 25.
    # We want to score items by their "value density" considering all constraints.
    # An item is good if it has high prize and low weight across constraints.
    
    item_prizes = prizes.sum(axis=0)  # Shape (100,)
    
    # Bottleneck slack: min_c (25 - w_ic) / w_ic
    # This measures how much "room" is left in the tightest constraint for this item.
    # A higher slack means the item fits more easily into the tightest constraint.
    capacity = 25.0
    # Avoid division by zero in weights
    safe_weights = np.maximum(weights, 1e-9)
    slack = (capacity - safe_weights) / safe_weights
    
    # The minimum slack across all constraints is the bottleneck slack
    bottleneck_slack = slack.min(axis=0)
    
    # Multiply by total prize to get a score that prioritizes high-value items
    # that also fit easily into the tightest constraints.
    heuristic = item_prizes * bottleneck_slack
    
    # Ensure positive values
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
