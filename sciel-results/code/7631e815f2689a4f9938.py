import numpy as np

def heuristics(prizes, weights):
    """
    Computes attractiveness scores using a constraint-specific scarcity premium heuristic.
    
    For each item, identifies the most binding constraint (where its weight is the largest 
    fraction of total weight) and scales its prize-to-weight ratio by the inverse of the 
    remaining capacity fraction for that specific constraint.
    """
    # Ensure inputs are float arrays
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    eps = 1e-9
    capacity = 25.0  # Fixed capacity per constraint
    
    # Calculate total weight per item across all constraints
    total_weights = np.sum(weights, axis=0)  # (100,)
    
    # Calculate total prize per item
    total_prizes = np.sum(prizes, axis=0)  # (100,)
    
    # Calculate the fraction of each constraint's weight for each item
    # weight_fraction[k, j] = weights[k, j] / total_weights[j]
    weight_fractions = weights / (total_weights + eps)  # (5, 100)
    
    # Identify the most binding constraint for each item
    # The constraint where the item's weight is the largest fraction of its total weight
    most_binding_constraint = np.argmax(weight_fractions, axis=0)  # (100,)
    
    # For each item, get the weight in its most binding constraint
    # We need to index weights with most_binding_constraint
    most_binding_weights = weights[most_binding_constraint, np.arange(len(most_binding_constraint))]  # (100,)
    
    # Calculate remaining capacity fraction for the most binding constraint
    # remaining_fraction = (capacity - most_binding_weight) / capacity
    remaining_fractions = (capacity - most_binding_weights) / capacity  # (100,)
    
    # Avoid division by zero in the premium calculation
    remaining_fractions = np.maximum(remaining_fractions, eps)
    
    # Calculate base prize-to-weight ratio for each item
    prize_to_weight_ratio = total_prizes / (total_weights + eps)  # (100,)
    
    # Apply scarcity premium: scale by inverse of remaining capacity fraction
    # Items with less remaining capacity in their most binding constraint get higher scores
    scarcity_premium = 1.0 / remaining_fractions  # (100,)
    
    # Final heuristic score
    scores = prize_to_weight_ratio * scarcity_premium
    
    # Ensure all scores are positive and finite
    scores = np.where(np.isfinite(scores) & (scores > 0), scores, 1e-9)
    
    return scores
