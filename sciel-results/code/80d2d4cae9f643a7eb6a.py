"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Probability of Improvement (PI) acquisition function.
    
    Uses PI instead of EI to prioritize points with high probability of improving
    the current best, normalized by the evaluation cost. This is a structural
    change from the Expected Improvement baseline, favoring reliability over
    expected magnitude of improvement, which can be beneficial in cost-aware
    settings to avoid high-cost "lottery ticket" evaluations.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Probability of Improvement
    # We want P(y < f_best) where y ~ N(mu, sigma^2)
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Cost-aware score: higher is better
    # Divide by cost to penalize expensive evaluations
    # cost is guaranteed to be >= 1 based on problem description
    scores = pi / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
