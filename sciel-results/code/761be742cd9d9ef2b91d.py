"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # prizes: (5, 100), weights: (5, 100)
    
    # Total prize per item (sum across all 5 prize dimensions)
    total_prizes = np.sum(prizes, axis=0)  # (100,)
    
    # Total weight per constraint (to estimate scarcity/tightness)
    total_weights_per_constraint = np.sum(weights, axis=1)  # (5,)
    
    # Avoid division by zero
    total_weights_per_constraint = np.where(
        total_weights_per_constraint > 1e-12, 
        total_weights_per_constraint, 
        1.0
    )
    
    # Normalized weight: how much of each constraint's total this item consumes
    # This captures constraint scarcity - items that consume a large fraction 
    # of a tight constraint are more costly
    weight_normalized = weights / total_weights_per_constraint[:, np.newaxis]  # (5, 100)
    
    # Total normalized cost per item: sum of fractional consumption across all constraints
    normalized_cost = np.sum(weight_normalized, axis=0)  # (100,)
    
    # Heuristic: prize density relative to normalized resource cost
    # Higher means better value per unit of scarce resource
    heuristic = total_prizes / (normalized_cost + 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
