"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Spectral clustering / variance-minimization approach.
# Goal: minimize the variance of remaining capacities across all open bins.
# This encourages a homogeneous set of gaps that are statistically more likely
# to be filled by future items, reducing entropy of the gap distribution.
#
# For each candidate bin i (with enough capacity):
#   - If we place the item in bin i, its remaining capacity becomes R_i = bins[i] - item
#   - All other bins keep their current remaining capacity
#   - Compute the variance of the resulting vector of remaining capacities
#   - Score = -variance (we want to minimize variance, so negate for max-selection)
#
# This is a structural change from the hybrid BF/WF approach. Instead of a
# simple per-bin heuristic, we consider the global effect on the distribution
# of gaps.
#
# Key insight: placing an item in a bin changes that bin's gap from bins[i] to
# bins[i] - item. The variance of the full vector changes based on how far
# bins[i] is from the current mean, and how the new value R_i relates to the
# new mean.
#
# For efficiency with up to ~1000+ bins, we precompute:
#   - sum_bins = sum of all bin remaining capacities
#   - sum_sq = sum of squares of all bin remaining capacities
#   - n = number of bins
#   - mean = sum_bins / n
#   - var = sum_sq / n - mean^2
#
# When we place item in bin j:
#   - new_sum = sum_bins - bins[j] + (bins[j] - item) = sum_bins - item
#   - new_sum_sq = sum_sq - bins[j]^2 + (bins[j] - item)^2
#   - new_mean = new_sum / n
#   - new_var = new_sum_sq / n - new_mean^2
#   - score_j = -new_var
#
# This is O(n) per call (one pass to compute sums, one pass to compute scores).
# For n up to a few thousand, this is well within the time budget.

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    n = len(bins)
    scores = np.full(n, -1e9)
    
    valid = bins >= item
    
    if not np.any(valid):
        return scores
    
    # Precompute global statistics
    sum_bins = np.sum(bins)
    sum_sq = np.sum(bins * bins)
    n_float = float(n)
    
    # New sum when we subtract item from one bin
    new_sum = sum_bins - item
    new_mean = new_sum / n_float
    
    # For each valid bin j, compute the resulting variance
    # new_sum_sq_j = sum_sq - bins[j]^2 + (bins[j] - item)^2
    #              = sum_sq - bins[j]^2 + bins[j]^2 - 2*bins[j]*item + item^2
    #              = sum_sq - 2*bins[j]*item + item^2
    
    valid_indices = np.where(valid)[0]
    valid_bins = bins[valid_indices]
    
    # Compute new_sum_sq for each valid bin
    new_sum_sq = sum_sq - 2.0 * valid_bins * item + item * item
    new_var = new_sum_sq / n_float - new_mean * new_mean
    
    # Score is negative variance (we maximize score = minimize variance)
    scores[valid_indices] = -new_var
    
    return scores
# EVOLVE-BLOCK-END
