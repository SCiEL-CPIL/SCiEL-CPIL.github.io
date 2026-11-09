import numpy as np


def heuristics(prizes, weights):
    # Use max weight across constraints as the effective weight
    # This penalizes items that are heavy in any single constraint
    max_weight = weights.max(axis=0)
    # Prize-to-weight ratio, squared to emphasize high-value items
    heuristic = (prizes / (max_weight + 1e-9)) ** 2
    # Ensure positive values
    heuristic = np.maximum(heuristic, 1e-9)
    return heuristic
