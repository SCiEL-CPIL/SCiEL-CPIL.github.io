import numpy as np
from scipy.stats import norm
import time

# Track start time for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 6d.
    Uses Expected Improvement with adaptive exploration weighting.
    
    Parameters:
    -----------
    mu : array of shape (N,)
        Mean of the Gaussian Process posterior at N candidate points.
    sigma : array of shape (N,)
        Standard deviation of the Gaussian Process posterior at N candidate points.
    f_best : float
        The lowest observed function value so far.
        
    Returns:
    --------
    scores : array of shape (N,)
        Higher scores indicate points with higher potential for improvement (lower true function value).
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: return zeros if we're too close to the 330s limit
    elapsed = time.time() - _start_time
    if elapsed > 320:
        return np.zeros_like(mu)
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Exploration parameter - slightly larger to encourage more exploration in 6D
    xi = 0.02
    
    # Calculate the z-score for the expected improvement
    z = (f_best - mu - xi) / sigma_safe
    
    # Compute Expected Improvement
    # EI = (f_best - mu - xi) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive exploration: weight sigma by how uncertain we are relative to the
    # expected improvement. Points with high sigma but low EI get a bigger boost.
    # This focuses exploration where it matters most.
    # Use a soft threshold: boost is proportional to sigma but dampened where EI is already high
    ei_max = np.max(ei)
    if ei_max > 1e-10:
        # Normalize EI to [0, 1] range for weighting
        ei_norm = np.clip(ei / ei_max, 0, 1)
        # Where EI is low, we want more exploration; where EI is high, less
        # Weight = base_weight * (1 - ei_norm * 0.5) gives between 0.5x and 1x of base
        base_weight = 0.15
        adaptive_weight = base_weight * (1.0 - 0.5 * ei_norm)
        ucb_boost = adaptive_weight * sigma_safe
    else:
        # If all EI values are near zero, use uniform exploration weight
        ucb_boost = 0.15 * sigma_safe
    
    # Combine EI and adaptive UCB boost
    scores = ei + ucb_boost
    
    # Ensure all returned scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
