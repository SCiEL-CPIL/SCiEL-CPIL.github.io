"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Pre-compute Weibull survival function values for scoring.
# The task uses Weibull sequences; we approximate the expected opportunity cost
# of holding remaining capacity as the integral of the survival function.
# For Weibull with shape k and scale lambda, survival S(x) = exp(-(x/lambda)^k)
# We use a reasonable default: k=1.5, lambda=50 (items in [0,100] capacity)
# The integral from r to inf of S(x) dx approximates expected waste.

# Pre-compute a lookup table for the expected waste as a function of remainder.
# We use numerical integration for remainders in [0, 100].
_K = 1.5
_LAMBDA = 50.0

# Build a lookup table for expected waste: E[r] = integral_r^inf exp(-(x/lambda)^k) dx
# This is related to the incomplete gamma function, but we can precompute numerically.
_N_TABLE = 10000
_rem_table = np.linspace(0, 100, _N_TABLE)
# Compute survival integral numerically using scipy if available, else simple integration
try:
    from scipy.special import gammainc, gamma as gamma_fn
    # integral_r^inf exp(-(x/lambda)^k) dx = lambda * gamma(1/k) * gammaincc(1/k, (r/lambda)^k)
    # where gammaincc is the regularized upper incomplete gamma
    _shape = 1.0 / _K
    _waste_table = _LAMBDA * gamma_fn(_shape) * (1.0 - gammainc(_shape, (_rem_table / _LAMBDA) ** _K))
except ImportError:
    # Fallback: numerical integration
    _waste_table = np.zeros(_N_TABLE)
    for i, r in enumerate(_rem_table):
        if r >= 100:
            _waste_table[i] = 0.0
        else:
            # Simple trapezoidal integration
            _x = np.linspace(r, 100 + 20 * _LAMBDA, 1000)
            _y = np.exp(-(_x / _LAMBDA) ** _K)
            _waste_table[i] = np.trapz(_y, _x)

# Normalize the waste table to [0, 1] for scoring
_waste_max = np.max(_waste_table)
if _waste_max > 0:
    _waste_table = _waste_table / _waste_max


def _expected_waste(remainder: np.ndarray) -> np.ndarray:
    """Look up expected waste for given remainders."""
    # Clip remainders to [0, 100]
    r = np.clip(remainder, 0, 100)
    # Interpolate from the precomputed table
    idx = np.clip((r / 100.0) * (_N_TABLE - 1), 0, _N_TABLE - 1).astype(int)
    # Linear interpolation
    frac = (r / 100.0) * (_N_TABLE - 1) - idx
    idx_next = np.clip(idx + 1, 0, _N_TABLE - 1)
    return _waste_table[idx] * (1 - frac) + _waste_table[idx_next] * frac


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin by negative expected waste of the remainder after placing item.
    Higher score = less expected waste = better bin.
    """
    remaining = bins - item
    valid = remaining >= 0
    if not np.any(valid):
        return np.full(len(bins), -np.inf)
    
    # Expected waste of the remaining capacity
    waste = _expected_waste(remaining[valid])
    
    # Score: negative waste (we want to minimize waste, so maximize score)
    # Add a small tiebreaker: prefer bins that are more filled (less total waste)
    # Actually, let's just use negative waste directly
    scores = np.full(len(bins), -np.inf)
    scores[valid] = -waste
    
    return scores
# EVOLVE-BLOCK-END
