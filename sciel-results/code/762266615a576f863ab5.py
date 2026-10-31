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
# We calculate the expected size of the next item.
# For Weibull(k, lambda), E[X] = lambda * Gamma(1 + 1/k)
_k = 2.0
_lambda_scale = 25.0
_EXPECTED_NEXT_ITEM = _lambda_scale * scipy.special.gamma(1 + 1/_k)

CAPACITY = 100.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Combinatorial Auction via Adversarial Robustness.
    
    We model the placement as a bid where the "cost" is the probability that the
    resulting remainder becomes dead waste. We use the Weibull survival function
    S(r) = exp(-(r/lambda)^k) to estimate the likelihood that the remainder r
    is too large to be filled by future items (i.e., it becomes dead space).
    
    However, simple Best-Fit (minimizing r) is often strong. To balance this,
    we use a composite score that penalizes:
    1. Large remainders (Worst Fit behavior) which waste capacity.
    2. "Dead zone" remainders where S(r) is high (unlikely to be filled).
    
    The score is designed to maximize the "usefulness" of the remainder.
    A remainder is useful if it is small enough to be filled (low r) or if it
    is large enough to be part of a consolidation strategy (but we don't know
    the future).
    
    Let's use the insight from Inspiration 1: Score = - (r * S(r)).
    This term represents the "Expected Dead Space" if we assume the next item
    is the only one that can fit. It creates a valley at r=0 and r->infinity,
    and a peak (bad score) in the middle. This encourages either perfect fits
    or very large remainders (which can be consolidated later), avoiding the
    "dead zone" in the middle.
    
    We also add a small linear penalty for r to encourage consolidation (Best Fit).
    Score = - (r * S(r) + alpha * r)
    
    Where:
    - S(r) = exp(-(r/lambda)^k)
    - alpha is a small weight for linear consolidation.
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
    # Clip r to avoid numerical issues, though r >= 0
    r = feasible_gaps
    # To avoid overflow in (r/lambda)^k for large r, we can clip the exponent
    # (r/lambda)^k can be large. exp(-large) -> 0.
    # Let's compute the argument safely.
    arg = (r / _lambda_scale) ** _k
    # Clip arg to prevent underflow issues in exp, though exp(-1000) is 0 anyway.
    # But if arg is huge, it's fine. If arg is small, exp(-arg) is close to 1.
    S_r = np.exp(-arg)
    
    # Dead Space term: r * S(r)
    dead_space = r * S_r
    
    # Linear consolidation penalty
    alpha = 0.1
    linear_penalty = alpha * r
    
    # Total penalty
    penalty = dead_space + linear_penalty
    
    # Score is negative penalty (we want to minimize penalty)
    final_scores = -penalty
    
    scores[feasible] = final_scores
    
    return scores
# EVOLVE-BLOCK-END
