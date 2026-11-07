"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    pair = demands[:, None] + demands[None, :]
    valid = pair <= capacity
    # Waste compatibility: attractiveness is inversely proportional to squared waste
    # waste = capacity - (demands[i] + demands[j])
    # When waste is small (tight fit), attractiveness is high
    # When waste is large, attractiveness is low
    waste = capacity - pair
    # For valid pairs, compute inverse squared waste with a small epsilon to avoid division by zero
    # Use 1 / (waste + eps)^2 to create a smooth potential field
    eps = 1.0
    inv_sq_waste = 1.0 / (waste + eps) ** 2
    # For invalid pairs (sum > capacity), set to 0
    score = np.where(valid, inv_sq_waste, 0.0)
    # Ensure all entries are positive (minimum 1e-9)
    score = np.maximum(score, 1e-9)
    return score
# EVOLVE-BLOCK-END
