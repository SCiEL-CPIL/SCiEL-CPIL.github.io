"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill

# Pre-compute Weibull CDF lookup table for fast evaluation
# Weibull distribution parameters (typical for bin packing benchmarks)
_WEIBULL_SHAPE = 1.5
_WEIBULL_SCALE = 20.0
_TABLE_SIZE = 10001
_CAPACITY = 100.0
_table_x = np.linspace(0, _CAPACITY, _TABLE_SIZE)
# CDF of Weibull: 1 - exp(-(x/scale)^shape)
_table_cdf = 1.0 - np.exp(-((_table_x / _WEIBULL_SCALE) ** _WEIBULL_SHAPE))
_table_survival = 1.0 - _table_cdf  # S(x) = P(X > x)

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Strategy: Weibull Survival-Based Best Fit.
    We model the packing as a stochastic inventory system where the cost of a
    remainder is the expected opportunity cost of holding that capacity. This is
    approximated by the integral of the Weibull survival function from the
    remainder to infinity, which quantifies the probability that the space
    remains unused by any future item. Score each valid bin by the negative of
    this expected waste.
    """
    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE:
        # Fallback to simple Best Fit if time is running out
        return -(bins - item)

    remaining_after = bins - item
    feasible = remaining_after >= 0
    
    if not np.any(feasible):
        return np.full(len(bins), -np.inf)
        
    scores = np.full(len(bins), -np.inf)
    
    # Calculate scores for feasible bins
    rem = remaining_after[feasible]
    
    if item > 0:
        # Use the pre-computed survival table
        # Map remainder to table index
        indices = np.clip((rem / _CAPACITY * (_TABLE_SIZE - 1)).astype(int), 0, _TABLE_SIZE - 1)
        survival_values = _table_survival[indices]
        
        # The expected waste is proportional to the integral of survival from rem to inf
        # For Weibull, E[X|X>rem] = rem + scale * Gamma(1+1/shape) * (rem/scale)^(-1/shape) * exp(-(rem/scale)^shape)
        # But simpler: just use the survival value as a proxy for "how much waste this remainder will cause"
        # Higher survival means more likely to be wasted, so we want to minimize it
        # Score = -survival (higher score = lower survival = better)
        scores[feasible] = -survival_values
    else:
        # If item is 0, any bin is fine, but prefer smallest remaining (Best Fit)
        scores[feasible] = -rem
        
    return scores
# EVOLVE-BLOCK-END
