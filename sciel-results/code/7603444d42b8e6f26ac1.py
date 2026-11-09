"""Acquisition function for Bayesian optimization of goldstein price 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for time management
_start_time = None


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling based acquisition function.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean of the GP at candidate points.
    sigma : ndarray, shape (2000,)
        Posterior standard deviation of the GP at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Thompson Sampling scores. Higher scores indicate better candidates.
    """
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    # Thompson Sampling: sample from the posterior
    # We want to minimize, so we sample from N(mu, sigma^2) and pick the lowest
    # But the evaluator picks the HIGHEST score, so we negate the sample
    
    # Ensure sigma is positive and handle edge cases
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Use a deterministic seed based on the current call to ensure reproducibility
    # but vary it slightly to get different samples each time
    # Actually, for Thompson Sampling we want random samples, but we need to be careful
    # about the time budget
    
    # Check time budget - if we're running low, fall back to a simpler strategy
    elapsed = time.time() - _start_time
    if elapsed > 300:
        # Fallback: use negative expected improvement as a safe default
        z = (f_best - mu) / (sigma_safe + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return ei
    
    # Thompson Sampling: sample from posterior and return negative (since we want to maximize score for minimization)
    # The evaluator will pick the candidate with the highest score
    # So we want high scores for candidates that are likely to have low function values
    
    # Sample from N(mu, sigma^2)
    # Use np.random with a fixed seed per call? No, we want randomness for exploration
    # But we need to be careful about reproducibility across seeds
    
    # Actually, the task says "for each seed 0, 1 and 2 the evaluator starts from 5 uniform random points"
    # The randomness in Thompson Sampling should come from the GP posterior, which is already randomized
    # by the training data. We just need to sample from it.
    
    # Sample from the posterior
    sample = mu + sigma_safe * np.random.randn(len(mu))
    
    # We want to minimize, so lower sample values are better
    # The evaluator picks the HIGHEST score, so we return negative of the sample
    scores = -sample
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
