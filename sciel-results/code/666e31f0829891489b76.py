"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Heuristic based on Bottleneck Density with Normalization.
    
    Approach:
    1. Calculate total prize for each item.
    2. Determine the 'bottleneck weight' for each item, which is the maximum
       weight it consumes in any single constraint. This identifies the most
       restrictive dimension of the item.
    3. Compute raw score as Prize / Bottleneck Weight.
    4. Normalize scores to the range [1, 2] to prevent numerical dominance
       by extreme ratios and to allow pheromone to modulate selection probability
       effectively.
    """
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # 1. Total prize for each item (sum across constraints)
    total_prizes = prizes.sum(axis=0)
    
    # 2. Bottleneck weight: max weight in any constraint for each item
    # This captures the item's "size" in its most constraining dimension
    bottleneck_weights = weights.max(axis=0)
    
    # 3. Raw heuristic score: Prize per unit of bottleneck weight
    epsilon = 1e-9
    raw_scores = total_prizes / (bottleneck_weights + epsilon)
    
    # 4. Normalize to [1, 2]
    # This ensures all heuristics are positive and of similar magnitude,
    # which stabilizes the ACO probability calculation (tau * eta).
    max_raw = np.max(raw_scores)
    if max_raw > 0:
        normalized = raw_scores / max_raw
    else:
        normalized = np.ones_like(raw_scores)
        
    # Shift to [1, 2]
    scores = 1.0 + normalized
    
    return scores
# EVOLVE-BLOCK-END
