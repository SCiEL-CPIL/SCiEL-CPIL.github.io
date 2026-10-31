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

# Polarization parameter: how much to penalize "middle" remainders
# This comes from Inspiration 2
CAPACITY = 100.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Combined Probabilistic Best-Fit with Weibull Dead-Space Awareness.
    
    We combine three insights:
    1. Best-Fit minimizes immediate waste (gap = bins[i] - item).
    2. Probabilistic penalty discourages creating very small gaps that cannot
       accommodate future items (which have an expected size E[X]).
    3. Weibull Dead-Space penalty discourages "middle" remainders that are likely
       to become unusable waste, leveraging the survival function S(r) = exp(-(r/lambda)^k).
       The penalty is proportional to r * S(r), which peaks in the "dead zone" and
       vanishes for very small or very large r.
    
    Score = -gap * multiplier - dead_space_penalty
    
    Where:
    - multiplier = 1 + exp(-(gap - E[X]) / sigma) penalizes small gaps
    - dead_space_penalty = weight * (gap * exp(-(gap/lambda)^k))
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
    
    # Calculate the probabilistic multiplier
    # sigma controls the sharpness of the transition
    sigma = 5.0
    
    # Argument for exponential
    arg = (feasible_gaps - _EXPECTED_NEXT_ITEM) / sigma
    # Clip to avoid overflow/underflow issues
    arg = np.clip(arg, -10, 10)
    
    multiplier = 1.0 + np.exp(-arg)
    
    # Base score: negative gap scaled by multiplier (from current program)
    base_score = -feasible_gaps * multiplier
    
    # Weibull Dead-Space penalty (from Inspiration 1)
    # S(r) = exp(-(r/lambda)^k)
    # Dead space penalty = r * S(r)
    # This penalizes remainders that are in the "dead zone" where they are
    # large enough to be visible waste but small enough to be unlikely to fit
    # a future item (high S(r) relative to r? No, S(r) is high for small r).
    # Actually, for Weibull with k=2, lambda=25:
    # If r=10, S(10) = exp(-(10/25)^2) = exp(-0.16) ~ 0.85. Penalty = 10*0.85 = 8.5
    # If r=25, S(25) = exp(-1) ~ 0.37. Penalty = 25*0.37 = 9.25
    # If r=50, S(50) = exp(-4) ~ 0.018. Penalty = 50*0.018 = 0.9
    # If r=5, S(5) = exp(-0.04) ~ 0.96. Penalty = 5*0.96 = 4.8
    # The penalty peaks around r ~ lambda/sqrt(k) ? Let's just use the formula.
    # We want to MINIMIZE this penalty, so we subtract it from the score.
    
    # To avoid overflow/underflow in exp, we clip the argument
    ratio = feasible_gaps / _lambda_scale
    # Clip ratio to avoid extreme values
    ratio_clipped = np.clip(ratio, 0, 10)
    exp_term = np.exp(-ratio_clipped ** _k)
    
    dead_space_penalty = feasible_gaps * exp_term
    
    # Weight for dead space penalty
    # The base score is roughly -gap * multiplier. For gap=20, multiplier~2, base=-40.
    # Dead space penalty for gap=20 is ~ 20 * exp(-(20/25)^2) = 20 * exp(-0.64) ~ 20*0.53 ~ 10.6
    # A weight of 0.5 would give -5.3 penalty. This is significant.
    ds_weight = 0.5
    
    # Final score
    final_scores = base_score - ds_weight * dead_space_penalty
    
    scores[feasible] = final_scores
    
    return scores
# EVOLVE-BLOCK-END
