"""Cost-aware acquisition for Hartmann 6d Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard
_DEADLINE = 310.0
_start_time = time.time()


def _check_time():
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a cost-normalized combination of Expected Improvement (EI) and Lower Confidence Bound (LCB).
    """
    if _check_time():
        return np.ones(2000)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Calculate Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe

    # Numerical stability
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)

    ei = (f_best - mu) * cdf + sigma_safe * pdf

    # Lower Confidence Bound (LCB) for minimization
    # We want to minimize mu - beta * sigma, so the "score" for LCB is -(mu - beta * sigma) = beta * sigma - mu
    # But we want higher scores to be better.
    # Standard LCB for minimization: argmin (mu - beta * sigma)
    # So the value to maximize is: -mu + beta * sigma
    # Let's use beta = 2.0
    beta = 2.0
    lcb_score = beta * sigma_safe - mu

    # Combine EI and LCB
    # Normalize EI and LCB scores to [0, 1] roughly, or just combine them
    # EI is non-negative. LCB score can be negative.
    # Let's try a weighted sum.
    # Since EI is in objective units and LCB is in objective units, we can combine them directly.
    # However, EI tends to be larger when improvement is large, while LCB is large when uncertainty is high.
    
    # Alternative: Use UCB (Upper Confidence Bound) for maximization? No, we minimize.
    # Let's stick to EI as the primary driver, but add a small LCB component to encourage exploration.
    
    # Actually, let's try a different approach: Cost-aware UCB for minimization?
    # No, let's try to improve the EI calculation or the cost normalization.
    
    # What if we use a different cost normalization?
    # cost_safe is 1 + mean(x). Since x is in [0,1], cost is in [1, 2].
    # Dividing by cost penalizes high-cost points.
    
    # Let's try a simple combination:
    # score = ei / cost_safe
    # This is the current implementation.
    
    # Let's try to add a small exploration bonus based on sigma.
    # score = (ei + 0.1 * sigma_safe) / cost_safe
    
    # Or maybe just use LCB?
    # score = lcb_score / cost_safe
    
    # Let's try a hybrid:
    # score = (0.8 * ei + 0.2 * lcb_score) / cost_safe
    
    # Let's check the scale of lcb_score.
    # If mu is around f_best, lcb_score is around beta * sigma.
    # If mu is much lower than f_best, lcb_score is negative.
    # EI is always non-negative.
    
    # Let's try to normalize lcb_score to be non-negative.
    # lcb_norm = np.maximum(lcb_score, 0)
    
    # Let's try a different strategy:
    # Use EI but with a different beta-like parameter? No, EI doesn't have beta.
    
    # Let's try to use the fact that we have 2000 candidates.
    # Maybe we can select the top K candidates and then apply a secondary criterion?
    # But the function must return 2000 scores.
    
    # Let's try to adjust the EI calculation to be more sensitive to uncertainty.
    # This is essentially what LCB does.
    
    # Let's try: score = (ei + 0.5 * sigma_safe) / cost_safe
    # This adds an exploration bonus proportional to sigma.
    
    # Let's try: score = (ei * (1 + 0.1 * sigma_safe)) / cost_safe
    # This multiplies EI by a factor that increases with sigma.
    
    # Let's try a simple addition first.
    scores = (ei + 0.5 * sigma_safe) / cost_safe
    
    # Handle NaN/inf
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
