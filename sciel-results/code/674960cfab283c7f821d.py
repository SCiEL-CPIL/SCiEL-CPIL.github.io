"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.special

# Track start time for deadline guard
_START_TIME = time.time()
_DEADLINE_MARGIN = 10.0  # seconds before the 150s limit
_MAX_EVAL_TIME = 150.0 - _DEADLINE_MARGIN

# Weibull distribution parameters for the generated sequences
# The task mentions "five generated Weibull sequences" with capacity 100.
# A common setup for these tasks uses Weibull(k, lambda) where items fit into bins of size 100.
# Assuming a standard Weibull with shape k=2.0 and scale lambda=25.0
# This yields a mean of ~22 and median of ~19, which is reasonable for capacity 100.
_k = 2.0
_lambda_scale = 25.0
_EXPECTED_NEXT_ITEM = _lambda_scale * scipy.special.gamma(1 + 1/_k)

CAPACITY = 100.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Game-Theoretic Adversarial Robustness with Probabilistic Dead Space.
    
    We combine the structural idea of the current program (Dead Space metric r * S(r))
    with the adversarial robustness concept: we want to avoid creating remainders that are
    "trapped" -- large enough to be useless for average items but too small for large items.
    
    The adversarial regret is approximated by the expected size of the largest item that
    could fit into the remainder, but weighted by the probability that such an item actually
    arrives. If the remainder is very small, the regret is small (no big item can fit).
    If the remainder is very large, the regret is large (a big item could fit and be blocked
    if we use this bin for a small item now).
    
    However, a more direct adversarial interpretation for online packing is:
    What is the worst-case future item that could have been placed here but is now blocked?
    If we place the current item in a bin leaving remainder r, we block any future item
    of size > r. The "regret" is the probability-weighted size of items in (r, CAPACITY].
    
    Let's use a simpler, computationally efficient proxy for adversarial robustness:
    Penalize remainders that are in the "danger zone" where they are likely to be wasted
    because they are too small for the next significant item but too large to be ignored.
    
    We use the composite score:
    Score = - (alpha * r * S(r) + beta * r + gamma * AdversarialTerm(r))
    
    Where:
    - r * S(r) is the probabilistic dead space (from current program).
    - beta * r is the linear consolidation penalty (Best Fit bias).
    - AdversarialTerm(r) penalizes remainders that are close to the expected item size,
      as these are the most likely to be "trapped" if the next item is slightly larger.
      We use a Gaussian-like penalty centered at _EXPECTED_NEXT_ITEM.
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
    
    # Calculate the survival function S(r) = exp(-(r/lambda)^k)
    r = feasible_gaps
    arg = (r / _lambda_scale) ** _k
    S_r = np.exp(-arg)
    
    # Dead Space term: r * S(r)
    dead_space = r * S_r
    
    # Linear consolidation penalty
    alpha = 0.1
    linear_penalty = alpha * r
    
    # Adversarial Robustness Term:
    # Penalize remainders that are close to the expected item size.
    # If r is close to E[X], it's a "danger zone" because a slightly larger item won't fit,
    # but a slightly smaller item would have fit if we had consolidated more.
    # We use a Gaussian penalty centered at E[X] with sigma = 5.0.
    sigma = 5.0
    adversarial_penalty = np.exp(-((r - _EXPECTED_NEXT_ITEM) / sigma) ** 2)
    
    # Weight for adversarial penalty
    # The adversarial term is in [0, 1], so we need a weight comparable to the other terms.
    # For r ~ 20, dead_space ~ 20 * exp(-(20/25)^2) ~ 20 * 0.85 ~ 17.
    # linear_penalty ~ 0.1 * 20 = 2.
    # So a weight of 5.0 for the adversarial term (max value 5.0) is reasonable.
    adv_weight = 5.0
    
    # Total penalty
    penalty = dead_space + linear_penalty + adv_weight * adversarial_penalty
    
    # Score is negative penalty (we want to minimize penalty)
    final_scores = -penalty
    
    scores[feasible] = final_scores
    
    return scores
# EVOLVE-BLOCK-END
