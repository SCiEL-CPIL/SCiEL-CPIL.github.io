"""Thompson Sampling acquisition for Bayesian optimization of Ackley 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_DEADLINE = time.time() + 320.0  # 320s safety margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling: sample from the posterior predictive for each candidate,
    then return the negative sample as the score (higher = more promising).
    
    Parameters:
        mu: array of shape (2000,) - posterior mean in objective units
        sigma: array of shape (2000,) - posterior std in objective units
        f_best: float - best observed value so far
    
    Returns:
        scores: array of shape (2000,) - finite scores, higher means evaluate next
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: return a simple exploitation-based score
        return -mu.copy()
    
    n = mu.shape[0]
    
    # Ensure sigma is strictly positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw one sample from N(mu_i, sigma_i^2) for each candidate
    # Use standard normal draws
    z = np.random.standard_normal(n)
    samples = mu + sigma_safe * z
    
    # Score: negative of the sample (we want to minimize, so lower sample = higher score)
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
