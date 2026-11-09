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
    
    # Probability of Improvement: P(f(x) < f_best)
    pi_scores = norm.cdf(z)
    
    # Assign infinite score to points with positive probability of improvement
    # This forces aggressive exploration of regions with high uncertainty or low mean
    # We use a large finite value instead of np.inf to avoid numerical issues
    # but effectively treat these as "must evaluate" candidates
    large_value = 1e10
    
    # Points where mu < f_best have high PI, but we want to emphasize exploration
    # by giving very high scores to points where there's meaningful uncertainty
    # Combined approach: use PI but boost points with high sigma
    
    # Standard PI with a twist: emphasize points where uncertainty is high
    # and there's a reasonable chance of improvement
    z_clipped = np.clip(z, -10, 10)  # Avoid extreme values in cdf
    pi_scores = norm.cdf(z_clipped)
    
    # Add a bonus for high uncertainty to encourage exploration
    # The bonus should be significant enough to drive exploration
    sigma_normalized = sigma / (np.max(sigma) + 1e-9)
    exploration_bonus = sigma_normalized * 0.5
    
    # Combine: PI score plus exploration bonus
    # This ensures points with high uncertainty get boosted
    scores = pi_scores + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=large_value, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
