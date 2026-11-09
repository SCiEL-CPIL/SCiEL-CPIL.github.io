"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (PI) acquisition function.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        Mean of the Gaussian process posterior at the candidate points.
    sigma : array_like, shape (2000,)
        Standard deviation of the Gaussian process posterior at the candidate points.
    f_best : float
        The lowest observed value so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores for each candidate point. Higher scores indicate
        points that are more likely to be evaluated next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score for PI: z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues in cdf
    z_clipped = np.clip(z, -10, 10)
    
    # Probability of Improvement: P(f(x) < f_best)
    pi_scores = norm.cdf(z_clipped)
    
    # The directive suggests aggressive exploration.
    # Pure PI can be too greedy. We add a strong exploration component
    # weighted by the standard deviation to favor uncertain regions.
    # We normalize sigma to have a comparable scale to the PI score [0, 1].
    sigma_max = np.max(sigma)
    if sigma_max > 1e-9:
        sigma_norm = sigma / sigma_max
    else:
        sigma_norm = sigma
        
    # Increase exploration weight significantly to "aggressively explore"
    # as requested, moving away from the modest gains of EI.
    exploration_bonus = sigma_norm * 5.0
    
    scores = pi_scores + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
