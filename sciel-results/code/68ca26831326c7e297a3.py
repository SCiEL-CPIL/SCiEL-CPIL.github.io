"""Rewritten acquisition for Bayesian optimization of Branin 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidates.
    Uses a combination of Expected Improvement (EI) and a lower-confidence bound (LCB)
    to balance exploitation and exploration.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Posterior mean at each candidate.
    sigma : array-like, shape (2000,)
        Posterior standard deviation at each candidate.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher is better.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement (EI)
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Lower Confidence Bound (LCB) with kappa for exploration
    # Using a moderate kappa to balance exploration and exploitation
    kappa = 1.5
    lcb = mu - kappa * sigma_safe
    
    # Combine EI and LCB. 
    # EI is in objective units (positive when improvement is expected).
    # LCB is in objective units (lower is better, so we negate for "higher is better" scoring).
    # We normalize both to have comparable scales before combining.
    
    # Scale EI: typical EI values are on the order of sigma
    # Scale LCB: we want to favor low mu and high sigma
    # A simple combination: weight EI heavily but add some LCB-based exploration bonus
    
    # Alternative: Use UCB-like approach but for minimization
    # UCB = mu - kappa * sigma (we want to maximize this? No, for minimization we want to minimize mu + kappa*sigma)
    # So for "higher is better" score, we can use: score = -(mu + kappa * sigma)
    # But this ignores the current best. EI is more principled.
    
    # Let's use a weighted sum: primarily EI, but add a small exploration term
    # to avoid getting stuck in local minima.
    
    # Normalize EI to have mean 0 and std 1 (if possible)
    ei_mean = np.mean(ei)
    ei_std = np.std(ei)
    if ei_std > 1e-10:
        ei_norm = (ei - ei_mean) / ei_std
    else:
        ei_norm = ei
    
    # LCB score: lower mu and higher sigma is better
    # Convert to "higher is better" by negating
    lcb_score = -mu + kappa * sigma_safe
    
    # Normalize LCB score
    lcb_mean = np.mean(lcb_score)
    lcb_std = np.std(lcb_score)
    if lcb_std > 1e-10:
        lcb_norm = (lcb_score - lcb_mean) / lcb_std
    else:
        lcb_norm = lcb_score
    
    # Combine: 80% EI, 20% LCB
    scores = 0.8 * ei_norm + 0.2 * lcb_norm
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
