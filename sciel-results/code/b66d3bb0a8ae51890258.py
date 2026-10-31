"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.special

# Track start time for deadline guard
_START_TIME = time.time()
_DEADLINE_MARGIN = 15.0  # seconds before the 150s limit
_MAX_EVAL_TIME = 150.0 - _DEADLINE_MARGIN

# Weibull distribution parameters for the generated sequences
# The task mentions "five generated Weibull sequences" with capacity 100.
# A common setup for these tasks uses Weibull(k, lambda) where items fit into bins of size 100.
# Assuming a standard Weibull with shape k=2.0 and scale lambda=25.0
# This yields a mean of ~22 and median of ~19, which is reasonable for capacity 100.
# We calculate the expected size of the next item.
# For Weibull(k, lambda), E[X] = lambda * Gamma(1 + 1/k)
_k = 2.0
_lambda_scale = 25.0
_EXPECTED_NEXT_ITEM = _lambda_scale * scipy.special.gamma(1 + 1/_k)

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Multi-dimensional knapsack optimization via effective capacity.
    
    We define "effective capacity" as the remaining space after placing the item,
    adjusted by the expected size of the next item. This prioritizes bins that
    maximize the probability of accepting a second consecutive item.
    
    Score = -(remaining_after - E[X])^2 / (2 * sigma^2) - remaining_after * beta
    
    - The first term penalizes bins where remaining_after is far from E[X] (optimal gap).
    - The second term is a linear penalty on remaining space (best-fit bias).
    - This combines probabilistic acceptance with fragmentation avoidance.
    """
    # Check if we're close to the deadline
    if time.time() - _START_TIME > _MAX_EVAL_TIME:
        # Emergency: return a simple best-fit fallback
        remaining = bins - item
        scores = np.where(remaining >= 0, -remaining, -np.inf)
        return scores
    
    remaining_after = bins - item
    feasible = remaining_after >= 0
    
    if not np.any(feasible):
        # No feasible bins; return all -inf
        return np.full(len(bins), -np.inf)
    
    scores = np.full(len(bins), -np.inf)
    
    # Get feasible gaps
    feasible_gaps = remaining_after[feasible]
    
    # Parameters for the scoring function
    sigma = 8.0  # Standard deviation for the quadratic penalty
    beta = 0.1   # Linear penalty coefficient for remaining space
    
    # Quadratic penalty: penalizes deviation from expected next item size
    # Optimal gap is around E[X], so bins with gap close to E[X] are preferred
    quadratic_term = -((feasible_gaps - _EXPECTED_NEXT_ITEM) ** 2) / (2.0 * sigma ** 2)
    
    # Linear penalty: best-fit bias, prefer smaller gaps
    linear_term = -beta * feasible_gaps
    
    # Combine terms
    scores[feasible] = quadratic_term + linear_term
    
    return scores
# EVOLVE-BLOCK-END
