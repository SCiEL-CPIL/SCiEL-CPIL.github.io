"""Thompson Sampling acquisition for Hartmann 6d.

Structural change: Replaces deterministic EI/UCB scoring with Thompson Sampling.
Instead of computing a single score per candidate, we draw one sample from the
posterior predictive distribution for each candidate and use that sample as the
score. This naturally balances exploration and exploitation: candidates with
high uncertainty will sometimes receive very favorable samples, while candidates
with low uncertainty near the current best will consistently score well.

The key advantage over EI/UCB is that Thompson Sampling provides a principled
way to explore multiple promising regions simultaneously, which is crucial for
the highly multimodal Hartmann 6d landscape.
"""

import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Parameters:
    mu: (2000,) array of GP mean predictions.
    sigma: (2000,) array of GP standard deviations.
    f_best: float, the best (lowest) observed value so far.
    
    Returns:
    (2000,) array of scores. Higher scores indicate better candidates.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure finite and non-negative sigma
    sigma = np.maximum(sigma, 1e-12)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-12)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    n = len(mu)
    
    # Thompson Sampling: draw one sample from N(mu, sigma^2) for each candidate
    # The candidate with the highest sampled value is selected
    # Since we're minimizing, we want the lowest value, so we negate
    # Actually, for Thompson Sampling in minimization:
    # We sample f ~ N(mu, sigma^2) and pick the candidate with the lowest sample
    # But the acquisition score should be "higher is better"
    # So we return -sample (negated) so that lower f values give higher scores
    
    # Draw samples from the posterior predictive
    samples = mu + sigma * np.random.standard_normal(n)
    
    # For minimization, lower sampled values are better
    # So score = -sample (higher score = better)
    scores = -samples
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
