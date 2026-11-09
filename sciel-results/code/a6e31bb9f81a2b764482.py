"""
Rewritten program using Optimistic Posterior Sampling (Max-of-Samples) acquisition.
"""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Optimistic Posterior Sampling acquisition function.
    
    Approximates the acquisition score by drawing 50 latent function samples
    from the GP posterior and taking the maximum improvement over f_best
    across these samples for each candidate point. This acts as a stochastic
    Upper Confidence Bound, favoring points with high potential improvement.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : array, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n_samples = 50
    
    # Ensure sigma is strictly positive for valid sampling
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw n_samples independent standard normal noise vectors
    # Shape: (n_samples, 2000)
    z = np.random.standard_normal((n_samples, mu.shape[0]))
    
    # Construct posterior samples: f_s(x) = mu(x) + sigma(x) * z_s(x)
    # Broadcasting mu and sigma to match noise shape
    samples = mu[None, :] + sigma_safe[None, :] * z
    
    # For minimization, the best sample at each point is the minimum value.
    # The maximum improvement is f_best - min_sample, clamped to 0.
    min_samples = np.min(samples, axis=0)
    scores = np.maximum(0.0, f_best - min_samples)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
