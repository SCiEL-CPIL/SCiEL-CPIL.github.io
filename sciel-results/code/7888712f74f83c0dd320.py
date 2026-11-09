import numpy as np
import time

def heuristics(prizes, weights):
    """
    Returns one attractiveness score per item for the MCKP ACO solver.
    Uses prize divided by a soft-max of weights across all constraints.
    The soft-max smoothly approximates the bottleneck constraint,
    favoring items that are light in their most constraining dimension.
    """
    start_time = time.time()
    deadline = start_time + 300.0

    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # Soft-max of weights across constraints for each item
    # Smoothly approximates the maximum weight (bottleneck constraint)
    beta = 5.0
    max_w = np.max(weights, axis=0)
    shifted = weights - max_w[np.newaxis, :]
    exp_shifted = np.exp(beta * shifted)
    soft_max = max_w + np.log(np.sum(exp_shifted, axis=0)) / beta

    # Heuristic: prize / soft_max_weight
    heuristic = prizes / (soft_max + 1e-9)

    # Ensure no non-finite values
    heuristic = np.nan_to_num(heuristic, nan=1e-9, posinf=1e9, neginf=1e-9)

    return heuristic
