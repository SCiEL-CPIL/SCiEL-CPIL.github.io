"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import cumulative_trapezoid

# Precomputed constants
_CAPACITY = 100.0
# Weibull parameters (guesses from seed, kept as they were part of the improvement)
_SHAPE = 1.5
_SCALE = 27.0

# Precompute the "Expected Waste" cost curve C(R)
# C(R) = R - integral_0^R x * f(x) dx
# This represents the expected remaining space in the bin after placing the current item,
# assuming the next item follows the Weibull distribution.
# Minimizing this cost is a principled heuristic.

_R_MAX = 100.0
_STEP = 0.1
_R_GRID = np.arange(0, _R_MAX + _STEP, _STEP)

# Weibull PDF: f(x) = (k/scale) * (x/scale)^(k-1) * exp(-(x/scale)^k)
x = _R_GRID
k = _SHAPE
scale = _SCALE

# Calculate PDF
# Avoid division by zero or negative base issues (x is non-negative)
# For x=0, if k>1, pdf=0. If k=1, pdf=1/scale. If k<1, pdf=inf.
# Here k=1.5, so pdf(0)=0.
with np.errstate(divide='ignore', invalid='ignore'):
    pdf = (k / scale) * (x / scale) ** (k - 1) * np.exp(-((x / scale) ** k))
pdf[0] = 0.0 # Explicitly handle 0

# Calculate integral of x * f(x) from 0 to R
integrand = x * pdf
integral = cumulative_trapezoid(integrand, x, initial=0)

# Cost C(R) = R - integral
_C_GRID = _R_GRID - integral

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin with enough remaining capacity.
    Returns an array of scores. Higher score is better.
    Score is based on minimizing the expected waste of the remainder.
    """
    # 1. Filter bins with enough capacity
    valid = bins >= item
    if not np.any(valid):
        return np.full_like(bins, -np.inf, dtype=float)
    
    # 2. Calculate remainders for valid bins
    remainders = bins[valid] - item
    
    # 3. Compute score based on Expected Waste model
    # Clamp remainders to valid grid range
    r_clamped = np.clip(remainders, 0, _R_MAX)
    
    # Interpolate the cost C(R)
    c_vals = np.interp(r_clamped, _R_GRID, _C_GRID)
    
    # We want to minimize cost, so score = -cost
    scores = -c_vals
    
    # 4. Construct output array
    out_scores = np.full_like(bins, -np.inf, dtype=float)
    out_scores[valid] = scores
    
    return out_scores

# EVOLVE-BLOCK-END
