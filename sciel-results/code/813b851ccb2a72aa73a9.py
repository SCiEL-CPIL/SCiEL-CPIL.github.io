"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Returns one attractiveness score per item (larger is more attractive).
    
    Uses a marginal utility heuristic that estimates the opportunity cost of each item
    by considering the scarcity of each resource dimension.
    
    Args:
        prizes: array of shape (5, 100) - prize values for each constraint/item
        weights: array of shape (5, 100) - weight values for each constraint/item
    
    Returns:
        array of shape (100,) - heuristic scores for each item
    """
    # prizes: (5, 100), weights: (5, 100)
    n_constraints, n_items = prizes.shape
    
    # Total prize per item (sum across constraints)
    total_prizes = np.sum(prizes, axis=0)  # (100,)
    
    # Total weight per constraint (sum across items)
    total_weights_per_constraint = np.sum(weights, axis=1)  # (5,)
    
    # Capacity is 25 for all constraints
    capacity = 25.0
    
    # Scarcity factor for each constraint: how much of the capacity is used on average
    # Higher scarcity means the resource is more valuable/constrained
    scarcity_factors = total_weights_per_constraint / capacity  # (5,)
    
    # For each item, compute the weighted sum of its weights using scarcity factors
    # This gives an "effective weight" that accounts for which resources are scarcer
    # weights.T is (100, 5), scarcity_factors is (5,)
    effective_weights = np.dot(weights.T, scarcity_factors)  # (100,)
    
    # Also consider the minimum slack: for each item, what's the tightest constraint it hits?
    # An item that uses a lot of a scarce resource has higher opportunity cost
    # Compute per-constraint weight normalized by capacity
    weight_norm = weights / capacity  # (5, 100) - fraction of capacity used per constraint
    
    # For each item, find the maximum fraction of any single constraint it uses
    # Items that are "bottleneck" on a tight constraint are less attractive
    max_constraint_usage = np.max(weight_norm, axis=0)  # (100,)
    
    # Combine: high prize, low effective weight, low max constraint usage
    eps = 1e-9
    
    # Base heuristic: prize / effective weight
    base_heur = total_prizes / (effective_weights + eps)
    
    # Penalty for items that heavily use any single scarce constraint
    # Use a soft penalty: items with high max_constraint_usage get penalized
    # The penalty is stronger when the constraint is scarce
    # For each item, compute the sum of (weight_in_constraint * scarcity) for the binding constraint
    binding_penalty = np.max(weight_norm * scarcity_factors[:, np.newaxis], axis=0)  # (100,)
    
    # Final heuristic: reward high prize, penalize high opportunity cost
    # Use a multiplicative form to keep values positive
    heur = base_heur / (1.0 + binding_penalty)
    
    # Ensure all values are positive and finite
    heur = np.where(np.isfinite(heur) & (heur > 0), heur, 1e-9)
    
    return heur
# EVOLVE-BLOCK-END
