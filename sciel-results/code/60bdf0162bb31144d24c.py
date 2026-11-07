"""Rewritten program using Thompson Sampling for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np

# Global random state for Thompson Sampling
_thompson_rng = np.random.default_rng()


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition: sample from the posterior and return
    negative sampled values so that higher scores correspond to lower
    (better) function values.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Posterior mean at each candidate point.
    sigma : np.ndarray, shape (2000,)
        Posterior standard deviation at each candidate point.
    f_best : float
        Best (lowest) observed function value so far.
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Finite scores; higher means evaluate next.
    """
    global _thompson_rng
    
    # Ensure sigma is non-negative and handle edge cases
    sigma = np.maximum(sigma, 1e-10)
    
    # Draw a sample from N(mu, sigma^2) for each candidate
    samples = _thompson_rng.normal(loc=mu, scale=sigma, size=mu.shape)
    
    # Return negative samples so that argmax corresponds to argmin of samples
    # This means the candidate with the lowest sampled function value gets the highest score
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
# EVOLVE-BLOCK-END
