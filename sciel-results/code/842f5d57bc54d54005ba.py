"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Pre-compute Weibull survival function values for remainders in [0, 100]
# The Weibull distribution with shape k and scale lambda has CDF: 1 - exp(-(x/lambda)^k)
# Survival function: exp(-(x/lambda)^k)
# For bin packing with capacity 100, items are likely in [0, 100].
# We'll use a Weibull with reasonable parameters for item sizes.
# Let's assume items follow a Weibull distribution with k=2, lambda=50 (mean ~45)
# The expected waste for a remainder r is the probability that no future item fits in r,
# which is approximated by the integral of the survival function from r to infinity.
# But actually, the opportunity cost of holding remainder r is the probability that
# a future item of size s <= r arrives and could have been placed, but we can't because
# we're looking at this specific bin's remainder.
# 
# A simpler model: the expected wasted capacity in a bin with remainder r is
# E[max(0, r - S)] where S is the size of the next item, or more accurately,
# the probability that the remainder remains unused.
#
# For a Weibull(k, lambda) distribution, P(S > r) = exp(-(r/lambda)^k)
# The expected waste is roughly r * P(no item fits) + E[min(S, r)] * P(S fits)
# But a simpler surrogate: score by -S(r) where S is the survival function,
# meaning we prefer bins with smaller remainders (higher survival = more likely unused).
#
# Actually, for best-fit, we want to minimize the remainder. The Weibull survival
# function gives us a smooth way to weight the remainder.
# Let's use: score = -exp(-(remainder/lambda)^k)
# This gives lower scores (more negative) for larger remainders, so the item goes
# to the bin with the smallest remainder, similar to best-fit but with a smooth penalty.

# Pre-compute a lookup table for the scoring function
# Remainders can be in [0, 100]
_LAMBDA = 50.0
_K = 2.0

def _score_remainder(remainder):
    """Score a bin given its remaining capacity after placing the item."""
    # We want to prefer bins with smaller remainders.
    # Use the Weibull survival function as a smooth penalty.
    # score = -exp(-(remainder/lambda)^k)
    # For remainder=0, score = -1 (best)
    # For remainder=100, score = -exp(-(100/50)^2) = -exp(-4) ≈ -0.018
    # So smaller remainders get more negative scores, and we pick the highest score.
    # Wait, we want the highest-scoring bin to be the one with the smallest remainder.
    # So we should return -exp(-(remainder/lambda)^k), which is more negative for larger remainders.
    # Actually, let's think again. We want to MINIMIZE the remainder.
    # If we return -remainder, that's best-fit.
    # With Weibull: return -exp(-(remainder/lambda)^k)
    # For remainder=0: -exp(0) = -1
    # For remainder=50: -exp(-(50/50)^2) = -exp(-1) ≈ -0.368
    # For remainder=100: -exp(-(100/50)^2) = -exp(-4) ≈ -0.018
    # So the highest score is -0.018 (remainder=100) and lowest is -1 (remainder=0).
    # That's backwards! We want the bin with the smallest remainder to have the highest score.
    # So we should return exp(-(remainder/lambda)^k) without the negative, or
    # return -(-exp(-(remainder/lambda)^k)) = exp(-(remainder/lambda)^k)
    # For remainder=0: exp(0) = 1 (highest)
    # For remainder=50: exp(-1) ≈ 0.368
    # For remainder=100: exp(-4) ≈ 0.018 (lowest)
    # Yes! So score = exp(-(remainder/lambda)^k)
    return np.exp(-((remainder / _LAMBDA) ** _K))


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    We use a Weibull-based smooth best-fit: prefer bins that leave the least
    remaining capacity, with a smooth penalty based on the Weibull survival function.
    """
    # Bins that can fit the item
    remaining = bins - item
    valid_mask = remaining >= 0
    
    # For invalid bins, set score to -inf so they're never chosen
    scores = np.full_like(remaining, -np.inf, dtype=np.float64)
    
    if np.any(valid_mask):
        # Compute smooth best-fit score for valid bins
        # score = exp(-(remainder/lambda)^k)
        # This is higher for smaller remainders
        valid_remainders = remaining[valid_mask]
        scores[valid_mask] = np.exp(-((valid_remainders / _LAMBDA) ** _K))
    
    return scores
# EVOLVE-BLOCK-END
