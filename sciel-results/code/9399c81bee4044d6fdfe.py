"""Cost-aware acquisition for cosine8 8d Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level cache for time tracking
_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the 330s kill to stop heavy work


def _elapsed():
    return time.time() - _start_time


def _time_exhausted():
    return _elapsed() > (330.0 - _DEADLINE_MARGIN)


def acquisition(mu, sigma, f_best, cost):
    """
    Batch-aware Greedy Expected Improvement (GEI) with uncertainty-reduction penalty.
    
    For each candidate i, the score approximates:
      EI_i / cost_i  -  lambda * (expected posterior variance reduction across the set)
    
    We approximate the variance reduction term by noting that evaluating a point with
    low sigma (already certain) reduces global uncertainty less than one in a high-sigma
    region. A cheap proxy: penalize candidates whose sigma is small (already well-known)
    relative to the median sigma, and reward those in high-uncertainty regions.
    
    Additionally, we use a PoI (Probability of Improvement) component blended with EI
    for robustness, as observed in prior measurements (PoI gave 1.03 vs 1.00 for EI).
    """
    n = len(mu)
    
    # Safety: if time is running out, return a simple score quickly
    if _time_exhausted():
        # Fallback: simple EI/cost, computed fast
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        scores = ei / np.maximum(cost, 1e-6)
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Standard EI
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # PoI: probability that mu < f_best (i.e., improvement)
    # P(f(x) < f_best) where f(x) ~ N(mu, sigma^2)
    # = Phi((f_best - mu) / sigma)
    poi = norm.cdf(z)
    
    # Blend EI and PoI: EI captures magnitude, PoI is more robust to miscalibration
    # Weight: use EI primarily but add a PoI component for exploration robustness
    blended = ei + 0.5 * sigma_safe * poi
    
    # Uncertainty-reduction penalty (GEI approximation):
    # Evaluating a point in a region where sigma is already low provides less
    # information gain. We approximate the "value of information" by penalizing
    # candidates whose sigma is below the median (already well-characterized).
    # This encourages the algorithm to explore under-explored regions.
    
    sigma_median = np.median(sigma_safe)
    # Penalty: 0 when sigma >= median, grows as sigma shrinks below median
    # Use a smooth soft-min-like form: penalty = max(0, median - sigma) / median
    penalty_raw = np.maximum(0.0, sigma_median - sigma_safe) / max(sigma_median, 1e-8)
    # Scale penalty: lambda controls strength
    lambda_penalty = 0.3
    penalty = lambda_penalty * penalty_raw
    
    # Cost normalization: divide by cost (higher cost = lower score)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Final score: blended improvement per unit cost, minus uncertainty penalty
    scores = (blended - penalty * sigma_safe) / cost_safe
    
    # Ensure finite, non-negative-ish (allow negative but clip to avoid extremes)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    # Clip to reasonable range to avoid numerical blowup
    scores = np.clip(scores, -10.0, 1e6)
    
    return scores
# EVOLVE-BLOCK-END
