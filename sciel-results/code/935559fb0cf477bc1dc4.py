import numpy as np
import time

# Global state
_start_time = None

# Weibull distribution parameters for item sizes
# k=2.0 (shape), lambda=12.0 (scale)
# This gives mean ~10.6, with a reasonable tail
_WEIBULL_K = 2.0
_WEIBULL_LAMBDA = 12.0

# Pre-compute quantiles for adversarial regret calculation
# We want to find the maximum item size that fits in remainder r
# such that P(X <= r) is at some confidence level
# For adversarial robustness, we use a high quantile (e.g., 95th percentile)
# to estimate the "worst-case" future item that could fit
# This encourages remainders that can handle large future items
_WEIBULL_PERCENTILE = 0.95
# Pre-compute the threshold: items larger than this are "rare"
# Q(p) = lambda * (-ln(1-p))^(1/k)
import math
_WEIBULL_Q95 = _WEIBULL_LAMBDA * (-math.log(1 - _WEIBULL_PERCENTILE)) ** (1.0 / _WEIBULL_K)

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    global _start_time
    
    # Initialize timing if not started
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: keep 5s margin before 150s limit
    elapsed = time.time() - _start_time
    if elapsed > 145:
        # Fallback to simple Best Fit
        remaining = bins - item
        valid = remaining >= 0
        scores = np.full(len(bins), -np.inf)
        if np.any(valid):
            scores[valid] = -remaining[valid]
        return scores

    # Calculate remaining capacity
    remaining = bins - item
    
    # Identify valid bins (enough capacity)
    valid = remaining >= 0
    
    # Initialize scores with -inf for invalid bins
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid):
        return scores
        
    valid_indices = np.where(valid)[0]
    new_rems = remaining[valid]
    
    # Adversarial robustness scoring:
    # For each remainder r, compute the "regret" as the maximum future item
    # size that could fit into r, weighted by the probability that such a large
    # item actually arrives. This encourages creating remainders that are
    # robust against worst-case immediate future demand.
    #
    # Key insight: A remainder r is "robust" if it can accommodate a wide range
    # of future item sizes without becoming dead space. We define regret as:
    #   regret(r) = E[max(X, 0) | X <= r] * P(X <= r)
    # This captures both the likelihood of a future item fitting and the
    # potential waste if a very large item (close to r) arrives.
    #
    # For Weibull(k, lambda), the truncated expectation is complex, so we
    # approximate using the quantile function:
    #   regret(r) ≈ r * F(r) * (1 - F(r)/2)
    # where F(r) is the CDF. This peaks at intermediate r and is low at
    # both extremes, but we add a consolidation bias to prefer smaller r.
    
    # Compute CDF: F(r) = 1 - exp(-(r/lambda)^k)
    scaled = new_rems / _WEIBULL_LAMBDA
    cdf = 1.0 - np.exp(-np.power(scaled, _WEIBULL_K))
    
    # Adversarial regret: r * F(r) * (1 - F(r)/2)
    # This is high when r is moderate and F(r) is moderate
    # Low when r is small (F(r) small) or r is large (1-F(r)/2 small)
    regret = new_rems * cdf * (1.0 - cdf / 2.0)
    
    # Consolidation bias: prefer smaller remainders to pack bins tightly
    # This helps reduce the number of partially filled bins
    consolidation = 0.1 * new_rems
    
    # Total penalty to minimize
    penalty = regret + consolidation
    
    # Score is negative penalty (higher is better)
    scores_valid = -penalty
    
    # Small tie-breaker: prefer lower index for stability
    scores_valid -= np.arange(len(valid_indices)) * 1e-10
    
    scores[valid] = scores_valid
    
    return scores
