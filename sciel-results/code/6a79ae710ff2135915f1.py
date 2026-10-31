"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Pre-compute a lookup table for the expected dead space under Weibull distribution
# Weibull with shape k=1.5, scale lambda=30 (typical for bin packing benchmarks)
# The expected dead space for a remainder r is E[min(X, r)] where X ~ Weibull(k, lambda)
# For small r, E[min(X,r)] ≈ r * P(X > r) + E[X * 1{X <= r}]
# Actually, E[min(X,r)] = ∫₀^r P(X > t) dt = ∫₀^r exp(-(t/lambda)^k) dt

_K = 1.5  # Weibull shape
_LAMBDA = 30.0  # Weibull scale

# Pre-compute E[min(X, r)] for r in [0, 100] using numerical integration
_TABLE_SIZE = 1001
_r_grid = np.linspace(0, 100, _TABLE_SIZE)
# E[min(X,r)] = ∫₀^r exp(-(t/lambda)^k) dt
# Use cumulative trapezoidal integration
_integrand = np.exp(-(_r_grid / _LAMBDA) ** _K)
_E_min_table = np.zeros(_TABLE_SIZE)
_E_min_table[0] = 0.0
for i in range(1, _TABLE_SIZE):
    _E_min_table[i] = _E_min_table[i-1] + 0.5 * (_integrand[i-1] + _integrand[i]) * (_r_grid[i] - _r_grid[i-1])

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity using Expected Dead Space metric.
    
    For each feasible bin, compute the expected dead space after placing the item.
    The expected dead space for remainder r is E[min(X, r)] where X ~ Weibull.
    We prefer bins with lower expected dead space (higher score = -expected_dead).
    """
    bins = np.asarray(bins, dtype=np.float64)
    
    # Feasible bins: remaining capacity >= item
    feasible = bins >= item
    
    # For feasible bins, compute expected dead space
    remaining_after = np.maximum(bins - item, 0.0)
    
    # Clip to valid range for lookup
    remaining_clipped = np.clip(remaining_after, 0, 100)
    
    # Use linear interpolation on the dead space table
    idx_float = remaining_clipped / 100.0 * (_TABLE_SIZE - 1)
    idx_floor = np.floor(idx_float).astype(int)
    idx_ceil = np.minimum(idx_floor + 1, _TABLE_SIZE - 1)
    frac = idx_float - idx_floor
    
    expected_dead = _E_min_table[idx_floor] * (1.0 - frac) + _E_min_table[idx_ceil] * frac
    
    # Score = -expected_dead_space (lower dead space = higher score)
    score = -expected_dead
    
    # For infeasible bins, set score to -inf
    score[~feasible] = -np.inf
    
    return score
# EVOLVE-BLOCK-END
