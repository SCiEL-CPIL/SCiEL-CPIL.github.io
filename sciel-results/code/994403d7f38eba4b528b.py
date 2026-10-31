"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Pre-compute Weibull survival function lookup table
# Weibull distribution parameters for item sizes
# Based on typical Weibull sequences used in bin packing benchmarks
_WEIBULL_SHAPE = 2.0
_WEIBULL_SCALE = 50.0  # Scale to match typical item size distribution
_CAP = 100.0

# Pre-compute survival function S(x) = P(X > x) for Weibull
# S(x) = exp(-(x/scale)^shape)
# We'll use a lookup table for fast evaluation
_N_TABLE = 1001
_TABLE_X = np.linspace(0, _CAP, _N_TABLE)
_TABLE_S = np.exp(-(_TABLE_X / _WEIBULL_SCALE) ** _WEIBULL_SHAPE)

def _weibull_survival(x):
    """Compute Weibull survival function S(x) = P(X > x) using lookup table."""
    x = np.clip(x, 0, _CAP)
    indices = (x / _CAP * (_N_TABLE - 1)).astype(int)
    indices = np.clip(indices, 0, _N_TABLE - 1)
    # Linear interpolation
    frac = (x / _CAP * (_N_TABLE - 1)) - indices
    frac = np.clip(frac, 0, 1)
    s0 = _TABLE_S[indices]
    s1 = _TABLE_S[np.minimum(indices + 1, _N_TABLE - 1)]
    return s0 + frac * (s1 - s0)

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    # Only consider bins with enough capacity
    valid = remaining >= 0
    
    # Expected opportunity cost of holding remaining capacity
    # Cost = integral of survival function from remaining to infinity
    # Approximate as: remaining * S(remaining) (simplified)
    # Better: use the fact that E[min(X, R)] where X ~ Weibull
    # The waste is the expected unused capacity = ∫_R^∞ S(x) dx
    
    # Compute expected waste for each bin
    # For Weibull, ∫_R^∞ S(x) dx can be approximated
    # Using the identity: ∫_R^∞ S(x) dx = ∫_0^∞ P(X + R > t) dt - R
    # Simpler approximation: waste ≈ S(R) * (scale / shape) * R^(1-shape)
    # But let's use a simpler model: waste is proportional to remaining * S(remaining)
    
    # Compute survival function at remaining capacity
    s_remaining = _weibull_survival(remaining)
    
    # Expected waste: approximate as remaining * s_remaining
    # This captures: large remaining + high probability of no future item = high waste
    expected_waste = remaining * s_remaining
    
    # Score: negative expected waste (lower waste = higher score)
    score = -expected_waste
    
    # Add a small best-fit component to break ties
    # and encourage filling bins
    score = score - 0.1 * remaining
    
    # Invalidate bins that can't fit the item
    score = np.where(valid, score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
