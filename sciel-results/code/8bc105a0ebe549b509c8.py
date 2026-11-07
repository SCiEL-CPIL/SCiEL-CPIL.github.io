"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Uses a combination of Expected Improvement (EI) and a UCB-like exploration term.
    The key insight from the measurements is that UCB(1) outperformed EI (1.0 -> 1.538),
    suggesting that an explicit, moderate exploration term works better than EI's implicit
    exploration for this short-horizon, multimodal problem.
    
    We use a hybrid approach:
    - Base score: Expected Improvement (captures exploitation)
    - Add a moderate UCB exploration term to encourage exploration
    - Scale the exploration term to be comparable to EI values
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = more promising)
    """
    # Ensure we have finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-10)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB exploration term (beta=1, which was shown to work well)
    ucb = 1.0 * sigma
    
    # Scale EI to be comparable with UCB
    # EI values can be small when f_best is close to mu, so we normalize
    # Use a weighted combination: 0.7 * EI + 0.3 * UCB
    # This balances exploitation and exploration
    
    # Normalize EI by its scale
    ei_scale = np.mean(ei[ei > 0]) if np.any(ei > 0) else 1.0
    if ei_scale < 1e-10:
        ei_scale = 1.0
    
    # Scale UCB to be in similar range as EI
    ucb_scale = np.mean(sigma)
    
    # Combine: use a fixed weight that emphasizes exploration slightly
    # Based on the finding that UCB(1) improved over EI, weight UCB more
    score = 0.5 * ei / ei_scale + 0.5 * ucb / ucb_scale
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
