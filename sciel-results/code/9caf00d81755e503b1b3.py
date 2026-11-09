"""Acquisition function for Bayesian optimization of ackley 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Compute an acquisition score for 2000 candidate points.
    
    Uses a hybrid of Expected Improvement (EI) and Upper Confidence Bound (UCB),
    with adaptive weighting based on the current best value and uncertainty.
    
    Parameters:
    -----------
    mu : ndarray of shape (2000,)
        Mean of the Gaussian process posterior.
    sigma : ndarray of shape (2000,)
        Standard deviation of the Gaussian process posterior.
    f_best : float
        Best (lowest) observed function value so far.
    
    Returns:
    --------
    scores : ndarray of shape (2000,)
        Acquisition scores; higher means more likely to be evaluated next.
    """
    # Ensure sigma is positive to avoid division issues
    sigma = np.maximum(sigma, 1e-10)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Upper Confidence Bound (UCB) - for minimization, we use lower confidence bound
    # LCB = mu - kappa * sigma
    # We want to maximize the acquisition, so higher score = more promising
    # For minimization, promising means low mu and/or high sigma (exploration)
    # UCB for minimization: score = -mu + kappa * sigma
    # Or equivalently, we can use: score = mu - kappa * sigma (lower is better)
    # But since we want higher score to be better, we use: -mu + kappa * sigma
    
    # Adaptive kappa: start with more exploration early (when f_best is high),
    # reduce exploration as we converge (when f_best approaches 0)
    # Use a smooth transition based on f_best
    # When f_best is large (early), kappa is larger (more exploration)
    # When f_best is small (late), kappa is smaller (more exploitation)
    
    # Normalize f_best to get a progress indicator
    # Ackley 2d minimum is 0, typical initial values are around 1-30
    # Use a smooth decay: kappa = kappa_min + (kappa_max - kappa_min) * exp(-f_best / scale)
    kappa_max = 2.5
    kappa_min = 0.5
    scale = 5.0
    kappa = kappa_min + (kappa_max - kappa_min) * np.exp(-f_best / scale)
    
    # UCB score for minimization (higher is better)
    ucb_score = -mu + kappa * sigma
    
    # Normalize EI to be on a similar scale as UCB for blending
    # EI is always >= 0, UCB can be negative
    # Scale EI by a factor to make it competitive with UCB
    # The typical scale of mu and sigma in objective units
    # Use the median sigma as a reference scale
    sigma_ref = np.median(sigma) + 1e-10
    ei_scaled = ei / sigma_ref
    
    # Blend EI and UCB with adaptive weights
    # Early on (high f_best), weight UCB more for exploration
    # Late on (low f_best), weight EI more for exploitation
    # Use the same exponential decay for the blend
    w_ucb = 0.3 + 0.7 * np.exp(-f_best / scale)
    w_ei = 1.0 - w_ucb
    
    # Combined score
    scores = w_ei * ei_scaled + w_ucb * ucb_score
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
