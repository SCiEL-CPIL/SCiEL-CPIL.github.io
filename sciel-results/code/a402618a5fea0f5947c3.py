"""
Acquisition function for Bayesian optimization of Hartmann 3D.
Strategy: Thompson Sampling.
We sample a single function value from the posterior distribution N(mu, sigma^2)
for each candidate point and return these samples as scores.
This stochastic approach encourages exploration of regions with high uncertainty
without needing to tune a kappa parameter or explicitly penalize density,
often leading to better global convergence in multimodal functions.
"""

import numpy as np
import time

# Global deadline guard
_start_time = time.time()
_TIME_LIMIT = 325.0  # Safety margin before 330s limit

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition scores.
    
    Args:
        mu: array of shape (n,), posterior mean in objective units.
        sigma: array of shape (n,), posterior standard deviation in objective units.
        f_best: float, lowest observed value so far.
        
    Returns:
        scores: array of shape (n,), finite scores. Higher score = higher priority.
    """
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Ensure sigma is non-negative and finite
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma = np.maximum(sigma, 1e-12)
    
    # Check time
    if time.time() - _start_time > _TIME_LIMIT:
        # Fallback to simple LCB if running out of time
        return -mu + 1.5 * sigma

    # Thompson Sampling: Sample f ~ N(mu, sigma^2) for each candidate
    # Since we are minimizing, we want to maximize the sampled value?
    # Wait. The task says "higher meaning evaluate next".
    # The objective is minimization.
    # If we sample f_hat from N(mu, sigma^2), a low f_hat is good.
    # So we should return -f_hat as the score?
    # Or simply return the sample, but the evaluator picks the MAX score.
    # If I return f_hat, the max f_hat is the "worst" predicted value.
    # So I must return -f_hat to make "good" (low) values have high scores.
    
    # However, standard Thompson Sampling for minimization often uses:
    # Sample f ~ N(mu, sigma^2), then pick argmin f.
    # Since the interface requires returning scores where HIGHER is better,
    # we return -sample.
    
    # To ensure reproducibility is not required but stability is, we just draw random numbers.
    # Note: The evaluator calls this function in a loop. The randomness is inherent to TS.
    
    # Optimization: Use numpy's random generator for speed
    # We use the global numpy random state for simplicity, or a local one?
    # Using global np.random is fine.
    
    # Draw samples
    # z ~ N(0, 1)
    z = np.random.standard_normal(n)
    samples = mu + sigma * z
    
    # Score is -sample (because we want to maximize the score, and low function values are good)
    scores = -samples
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
