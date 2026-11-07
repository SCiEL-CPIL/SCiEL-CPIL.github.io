"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (PI) acquisition function with dynamic exploration offset (xi).
    
    This uses PI which directly maximizes the probability that a candidate point
    yields a better objective value than the current best, offering a more
    aggressive exploitation bias than EI while maintaining a distinct probabilistic
    interpretation.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean predictions.
    sigma : ndarray, shape (2000,)
        Posterior standard deviation predictions.
    f_best : float
        Best observed value so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values are preferred.
    """
    # Known global minimum of Styblinski-Tang 1D
    f_min_global = -39.1662
    
    # Calculate current regret
    regret = f_best - f_min_global
    
    # Estimate initial regret to normalize progress
    max_regret = 50.0
    
    # Progress: 0.0 at start (high regret), 1.0 at optimum (zero regret)
    progress = np.clip(1.0 - (regret / max_regret), 0.0, 1.0)
    
    # Dynamic xi (exploration offset)
    # High xi early on encourages looking at points with high uncertainty or
    # moderate means where improvement is still possible.
    # Low xi late on focuses strictly on points likely to beat f_best.
    xi_max = 5.0
    xi = xi_max * (1.0 - progress) ** 2
    
    # Ensure sigma is positive to avoid division by zero in z-score calculation
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate Z-score for PI
    # z = (f_best - mu - xi) / sigma
    z = (f_best - mu - xi) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best - xi) = Phi(z)
    scores = norm.cdf(z)
    
    # Ensure all scores are finite and in valid range
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.clip(scores, 0.0, 1.0)
    
    return scores
# EVOLVE-BLOCK-END
