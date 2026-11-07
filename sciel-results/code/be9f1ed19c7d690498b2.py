"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Computes attractiveness scores for items in a Multiple-Constraint Knapsack Problem.
    
    Uses a geometric mean of per-constraint efficiency ratios (prize_i / weight_ij),
    which balances value against the multiplicative cost of consuming resources across
    all dimensions simultaneously. This penalizes items with high variance in resource
    consumption more smoothly than max/min approaches.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin for 300s limit

    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # Check for immediate deadline breach (unlikely for O(N) but good practice)
    if time.time() > deadline:
        return np.ones(prizes.shape[1])

    # Calculate per-constraint efficiency ratios: prize / weight for each constraint
    # Shape: (num_constraints, num_items)
    epsilon = 1e-9
    ratios = prizes / (weights + epsilon)
    
    # Geometric mean: exp(mean(log(ratios)))
    # Clip ratios to avoid log(0) or log(negative)
    ratios_clipped = np.clip(ratios, epsilon, None)
    log_ratios = np.log(ratios_clipped)
    
    # Geometric mean across constraints (axis=0)
    geometric_mean = np.exp(np.mean(log_ratios, axis=0))
    
    # Ensure all values are positive and finite
    # The task states: "Heuristic entries that are not positive become 1e-9"
    # We handle non-finite values by replacing them with a small positive number
    # and ensuring positivity.
    
    # Replace NaN or Inf with 0, then set non-positive to 1e-9
    geometric_mean = np.where(np.isfinite(geometric_mean), geometric_mean, 0.0)
    geometric_mean = np.where(geometric_mean > 0, geometric_mean, 1e-9)
    
    return geometric_mean
# EVOLVE-BLOCK-END
