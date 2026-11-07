"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = time.time()


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of michalewicz 2d.
    Uses a combination of Expected Improvement (EI) and Probability of Improvement (PI)
    with adaptive exploration to balance exploitation and exploration.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the Gaussian process posterior at candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of the Gaussian process posterior at candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    # Deadline guard: return safe scores if approaching time limit
    elapsed = time.time() - _start_time
    if elapsed > 320:
        return np.zeros_like(mu)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute Probability of Improvement (PI) for minimization
    # PI = Phi(z) where z = (f_best - mu) / sigma
    pi = norm.cdf(z)
    
    # Adaptive exploration: blend EI and PI based on uncertainty
    # When sigma is large relative to the scale of f_best, favor PI for exploration
    # When sigma is small, favor EI for exploitation
    
    # Estimate scale from the data
    mu_scale = np.std(mu) + 1e-10
    sigma_mean = np.mean(sigma_safe)
    
    # Exploration weight: higher when uncertainty is high relative to function scale
    # Use a smooth transition based on the ratio of sigma to mu spread
    exploration_weight = np.clip(sigma_safe / (mu_scale + sigma_mean + 1e-10), 0.0, 1.0)
    
    # Normalize EI to [0, 1] for blending
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min + 1e-10
    ei_normalized = (ei - ei_min) / ei_range
    
    # Blend: score = (1 - w) * EI_normalized + w * PI
    # This gives more weight to PI (exploration) when uncertainty is high
    scores = (1.0 - exploration_weight) * ei_normalized + exploration_weight * pi
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
