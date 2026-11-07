"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # prizes: (5, 100), weights: (5, 100), all in [0, 1]
    # Total prize per item
    total_prize = prizes.sum(axis=0)  # (100,)
    
    # Calculate total weight per constraint
    # weights shape (5, 100), sum over items (axis=1) gives (5,)
    total_weights_per_constraint = weights.sum(axis=1)
    
    # Capacity is 25 for each constraint
    capacity = 25.0
    
    # Scarcity factor: how full is the constraint relative to capacity?
    # Higher scarcity -> higher dual price approximation
    scarcity = total_weights_per_constraint / capacity
    
    # Effective dual cost for each item: sum of (weight_in_constraint * scarcity_of_constraint)
    # weights.T shape (100, 5), scarcity shape (5,)
    # dot product gives (100,)
    dual_cost = np.dot(weights.T, scarcity)
    
    # Heuristic score: Prize / Dual Cost
    # Add epsilon to avoid division by zero
    scores = total_prize / (dual_cost + 1e-9)
    
    # Normalize so mean is ~1 to keep heuristic in same scale as pheromone
    mean_score = scores.mean()
    if mean_score > 0:
        scores = scores / mean_score
        
    return scores
# EVOLVE-BLOCK-END
