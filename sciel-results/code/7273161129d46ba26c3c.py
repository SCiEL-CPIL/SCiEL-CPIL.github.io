"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    Uses Kriging Believer strategy: score is -mu, but only for points that are
    top candidates according to UCB (mu - beta*sigma). This decouples exploration
    (via UCB) from exploitation (via -mu), avoiding selection of noise peaks.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points (objective units).
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points (objective units).
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Ensure inputs are numpy arrays for efficient vectorized operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety margin
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute UCB for minimization: we want to explore where mu - beta*sigma is low
    # (i.e., potentially low function value). For selection, we look at the top
    # candidates by UCB score (higher is better for selection in maximization sense).
    # UCB for minimization: score_ucb = -mu + beta*sigma (higher is more promising to explore)
    beta = 1.0
    ucb_scores = -mu + beta * sigma_safe
    
    # Select top K% of candidates by UCB score
    # This identifies promising exploration regions
    n = len(mu)
    k = max(1, int(0.1 * n))  # Top 10%
    
    # Get indices of top k UCB candidates
    top_k_indices = np.argpartition(ucb_scores, -k)[-k:]
    
    # Initialize scores with very low values
    scores = np.full(n, -1e10, dtype=np.float64)
    
    # For the top K UCB candidates, use Kriging Believer score: -mu
    # This means we exploit the low mean within the explored regions
    scores[top_k_indices] = -mu[top_k_indices]
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
