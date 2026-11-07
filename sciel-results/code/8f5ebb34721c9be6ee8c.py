"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix):
    """
    Compute an edge-attractiveness matrix for ACO on TSP.
    Uses a combination of inverse distance and a penalty for long edges
    to balance exploration and exploitation.
    """
    start_time = time.time()
    deadline = start_time + 5.0  # Safety margin, though this is very fast

    n = distance_matrix.shape[0]
    # Mask diagonal (self-loops) - they should not be attractive
    d = distance_matrix.copy()
    np.fill_diagonal(d, 1e9)

    # Base heuristic: inverse distance squared (stronger preference for short edges)
    # Add a small component that penalizes very long edges more aggressively
    # to help ants avoid crossing the square
    inv_d2 = 1.0 / (d + 1e-9)**2

    # Additional penalty for long edges (above median distance)
    median_d = np.median(d[d > 1e-8])
    long_penalty = np.where(d > median_d * 1.5, 0.5, 1.0)

    # Combine: base heuristic with a soft penalty for long edges
    eta = inv_d2 * long_penalty

    # Ensure all values are positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)

    # Check deadline (should never trigger for 50x50 matrices)
    if time.time() > deadline:
        # Fallback: simple inverse distance
        return 1.0 / (d + 1e-9)

    return eta
# EVOLVE-BLOCK-END
