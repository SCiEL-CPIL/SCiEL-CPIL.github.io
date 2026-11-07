# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 6d.
    
    Parameters:
    -----------
    mu : ndarray, shape (N,)
        Mean of the GP posterior at N candidate points.
    sigma : ndarray, shape (N,)
        Standard deviation of the GP posterior at N candidate points.
    f_best : float
        Best observed value so far (lowest).
    
    Returns:
    --------
    scores : ndarray, shape (N,)
        Acquisition scores; higher is better.
    """
    start_time = time.time()
    deadline = 330.0 # 300s limit + margin, but we should return much faster.
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Calculate UCB score for minimization:
    # We want to minimize f(x).
    # LCB = mu - beta * sigma. We want to pick the point with the lowest LCB.
    # Score = -LCB = -mu + beta * sigma. Higher score is better.
    beta = 1.5
    ucb_score = -mu + beta * sigma_safe
    
    # Ensure UCB scores are finite
    ucb_score = np.where(np.isfinite(ucb_score), ucb_score, 0.0)
    
    # Hybrid: Combine EI and UCB.
    # EI is in objective units. UCB is in objective units.
    # They should be comparable in scale.
    # We can average them or weight them.
    # Let's try a weighted sum.
    # If EI is very small (exploitation), UCB might dominate (exploration).
    # If UCB is high due to noise, EI might be low.
    
    # Normalize scores to [0, 1] range for stable combination?
    # This can be risky if the range is tiny.
    
    # Simple weighted sum
    w_ei = 0.5
    w_ucb = 0.5
    
    scores = w_ei * ei + w_ucb * ucb_score
    
    # Check time
    if time.time() > deadline - 1.0:
        # Return best effort
        return scores
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
