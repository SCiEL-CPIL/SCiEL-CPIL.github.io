"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Multi-sample Thompson Sampling acquisition function.
    
    mu: array of shape (2000,) - posterior mean at candidates
    sigma: array of shape (2000,) - posterior std at candidates
    f_best: float - best observed value so far
    
    Returns: array of shape (2000,) - scores, higher means evaluate next
    """
    n_candidates = len(mu)
    n_samples = 100
    
    # Generate random samples from the marginal posterior at each candidate
    # Shape: (n_samples, n_candidates)
    # Use a fresh RNG instance to ensure different random draws at each acquisition step
    rng = np.random.default_rng()
    samples = mu[None, :] + sigma[None, :] * rng.standard_normal((n_samples, n_candidates))
    
    # Multi-sample Thompson Sampling estimate:
    # For each candidate, compute the minimum value observed across the n_samples draws.
    # This approximates the lower tail of the predictive distribution.
    # We return the negative of this minimum, so that candidates with very low
    # sampled values (high potential for improvement) receive higher scores.
    min_samples = np.min(samples, axis=0)
    scores = -min_samples
    
    return scores
# EVOLVE-BLOCK-END
