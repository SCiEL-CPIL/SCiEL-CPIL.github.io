"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = time.time()
_TIME_LIMIT = 325.0  # safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for minimizing branin 2d.
    
    Uses a combination of Expected Improvement (EI) and a UCB exploration term
    with adaptive beta that decays over the course of optimization.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Mean of GP posterior at candidate points
    sigma : array of shape (2000,)
        Standard deviation of GP posterior at candidate points
    f_best : float
        Best (lowest) observed objective value
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores, higher means evaluate next
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        # Return a simple score to avoid timeout
        sigma_safe = np.maximum(sigma, 1e-9)
        return f_best - mu
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Estimate which iteration we're in based on the magnitude of uncertainty
    # Early: high sigma everywhere -> explore more
    # Late: low sigma -> exploit more
    
    # Use the median sigma as a proxy for exploration level
    med_sigma = np.median(sigma)
    
    # Adaptive beta: start high for exploration, decay to low for exploitation
    # Beta controls the UCB exploration term
    # When med_sigma is large (early), we want more exploration
    # When med_sigma is small (late), we want more exploitation
    
    # Scale beta based on typical uncertainty relative to f_best scale
    # Branin has minimum ~0.398, max ~302, so scale is roughly 300
    # Use a smooth decay: beta = beta_max * (med_sigma / (med_sigma + beta_scale))
    beta_max = 3.0
    beta_scale = 1.0  # controls how quickly beta decays
    
    beta = beta_max * (med_sigma / (med_sigma + beta_scale))
    # Clamp beta to reasonable range
    beta = np.clip(beta, 0.1, beta_max)
    
    # Compute Expected Improvement (EI) for minimization
    z_ei = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma * norm.pdf(z_ei)
    
    # Compute UCB (Lower Confidence Bound for minimization)
    # LCB = mu - beta * sigma (lower is better)
    # We want higher acquisition score = better, so:
    # UCB_score = -mu + beta * sigma
    ucb_score = -mu + beta * sigma
    
    # Combine EI and UCB
    # Normalize both to have similar scale
    # EI is in objective units, UCB_score is also in objective units
    # Weight: more EI when exploitation is beneficial, more UCB when exploring
    
    # Simple combination: weighted sum
    # When uncertainty is high, weight UCB more
    # When uncertainty is low, weight EI more
    w_ucb = beta / (beta + 1.0)
    w_ei = 1.0 - w_ucb
    
    scores = w_ei * ei + w_ucb * ucb_score
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
