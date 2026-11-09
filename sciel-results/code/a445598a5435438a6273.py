"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack.
    
    Strategy:
    - Use the maximum weight across constraints (the binding constraint) as the denominator
    - This focuses on the most restrictive constraint for each item
    - Apply a power of 2.5 to sharpen the distribution, strongly favoring high value-density items
    - This helps ACO ants converge faster on the best candidates
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Use max weight across constraints (the binding constraint)
    max_weight = weights.max(axis=0)
    
    # Base ratio: prize / max_weight
    # Add small epsilon to avoid division by zero
    ratio = prizes / (max_weight + 1e-9)
    
    # Apply power of 2.5 to sharpen distribution
    # This creates stronger differentiation between high and low value-density items
    scores = ratio ** 2.5
    
    # Ensure all values are positive (required by the task)
    scores = np.maximum(scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
