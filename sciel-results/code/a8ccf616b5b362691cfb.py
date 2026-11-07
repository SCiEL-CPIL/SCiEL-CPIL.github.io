"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # Handle case where prizes might be 2D (5, 100)
    if prizes.ndim == 2:
        prizes = prizes.mean(axis=0)
    
    capacity = 25.0
    
    # For each constraint c, compute total competing prize:
    # total_comp[c] = sum_j prizes[j] * weights[c, j]
    total_comp = weights @ prizes  # shape (5,)
    
    # For each item i, compute competitive pressure:
    # pressure[i] = sum_c (weights[c, i] / capacity) * total_comp[c]
    pressure = weights.T @ (total_comp / capacity)  # shape (100,)
    
    # Competitive exclusion score: prize discounted by competitive pressure
    score = prizes / (1.0 + pressure)
    
    # Ensure positive and finite values
    score = np.where(np.isfinite(score) & (score > 0), score, 1e-9)
    
    return score
# EVOLVE-BLOCK-END
