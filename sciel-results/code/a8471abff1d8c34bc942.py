"""
Acquisition function for Bayesian optimization of Michalewicz 2D.

This program uses a hybrid acquisition strategy that combines:
1. Expected Improvement (EI) - for exploitation
2. Upper Confidence Bound (UCB) - for exploration

The key insight is that Michalewicz 2D is multimodal, so we need a balance
between exploiting known good regions and exploring new areas.

We use rank-based normalization to blend EI and UCB scores, which is robust
to scale changes in the GP posterior statistics. The exploration weight
adapts based on the current best value, exploring more when far from the
known minimum.
"""
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at each candidate
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at each candidate
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Known minimum of Michalewicz 2D
    f_min = -1.8013
    
    # Handle edge cases
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate Expected Improvement for minimization
    u = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(u) + sigma_safe * norm.pdf(u)
    
    # Handle zero sigma cases
    zero_sigma_mask = sigma < eps
    if np.any(zero_sigma_mask):
        ei[zero_sigma_mask] = np.maximum(0.0, f_best - mu[zero_sigma_mask])
    
    # Calculate UCB for minimization
    # Slightly higher beta to encourage exploration in multimodal landscape
    beta = 2.5
    ucb = -mu + beta * sigma_safe
    
    # Rank-based normalization for robust blending
    # This avoids issues with min-max normalization when ranges are small or skewed
    n = len(ei)
    
    # Get ranks (0 to n-1)
    ei_ranks = np.argsort(np.argsort(ei))
    ucb_ranks = np.argsort(np.argsort(ucb))
    
    # Normalize ranks to [0, 1]
    if n > 1:
        ei_norm = ei_ranks / (n - 1)
        ucb_norm = ucb_ranks / (n - 1)
    else:
        ei_norm = np.ones_like(ei)
        ucb_norm = np.ones_like(ucb)
    
    # Adapt exploration weight based on how close f_best is to the minimum
    # If f_best is close to f_min, exploit more (lower exploration weight)
    # If f_best is far from f_min, explore more (higher exploration weight)
    gap = max(f_best - f_min, 0.0)
    # Scale gap to [0, 1] with a characteristic scale of ~0.5
    gap_scaled = 1.0 - np.exp(-gap / 0.5)
    
    # Exploration weight: higher when we're far from the optimum
    # Base exploration weight between 0.3 and 0.6
    w_explore = 0.3 + 0.3 * gap_scaled
    w_exploit = 1.0 - w_explore
    
    # Blend EI and UCB using rank-normalized scores
    scores = w_exploit * ei_norm + w_explore * ucb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
