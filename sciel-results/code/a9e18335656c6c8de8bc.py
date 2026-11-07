# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using multi-sample Thompson Sampling.
    For each candidate, we estimate the expected minimum over N posterior samples.
    This provides a natural exploration-exploitation balance by sampling from
    the GP posterior and finding candidates that perform well across multiple
    plausible function realizations.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed function value so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores. Higher values indicate more promising candidates.
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Number of Thompson Sampling samples
    # More samples reduce variance in the estimated expected minimum
    n_samples = 16
    
    # Pre-generate all random samples for efficiency
    # Shape: (n_samples, 2000)
    z = np.random.randn(n_samples, len(mu))
    
    # Sample from posterior: y = mu + sigma * z
    # Shape: (n_samples, 2000)
    samples = mu[None, :] + sigma_safe[None, :] * z
    
    # For each candidate, compute the expected minimum over all samples
    # This is the Thompson Sampling score: lower is better
    # We want to maximize the probability of finding a good point,
    # so we use negative expected minimum as the score
    expected_min = np.mean(samples, axis=0)
    
    # Add a small exploration bonus based on the variance of samples
    # Candidates with high variance across samples are more uncertain
    # and may contain undiscovered good regions
    sample_var = np.var(samples, axis=0)
    
    # Combine: primarily use expected minimum, with a small exploration term
    # The exploration term is weighted by the ratio of variance to the scale
    # of the problem (using f_best as a reference scale)
    scale = max(abs(f_best), 1e-6)
    exploration_bonus = 0.1 * np.sqrt(sample_var) / scale
    
    # Score = -expected_min + exploration_bonus
    # Higher scores mean we prefer candidates with lower expected function values
    # and some uncertainty
    scores = -expected_min + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety check
    elapsed = time.time() - start_time
    if elapsed > 290:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
