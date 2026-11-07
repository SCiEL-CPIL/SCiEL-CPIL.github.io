"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Return one attractiveness score per item for the ACO knapsack heuristic.
    
    The heuristic is designed to prioritize items that have high prize value
    relative to their most constraining weight dimension. We use:
      - total prize (sum across constraints) as the value signal
      - inverse of the maximum single-constraint weight as the cost signal
    
    This captures 'criticality': items that are high-value but don't heavily
    burden any single tight constraint are more attractive.
    
    We add a small epsilon to avoid division by zero and ensure all values
    are positive (the evaluator replaces non-positive with 1e-9 anyway).
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Total prize per item (sum across the 5 constraint rows)
    total_prize = prizes.sum(axis=0)
    
    # Maximum weight per item across constraints - identifies the tightest bottleneck
    max_weight = weights.max(axis=0)
    
    # Heuristic: prize scaled by inverse of max weight
    # Items with high prize and low max-weight are most attractive
    heuristic = total_prize / (max_weight + 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
