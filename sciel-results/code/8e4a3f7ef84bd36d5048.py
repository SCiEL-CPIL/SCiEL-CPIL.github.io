# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    with a UCB-like exploration term, normalized by cost.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best (lowest) observed value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled)).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Finite acquisition scores (higher = more desirable).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure no division by zero or negative costs
    cost = np.maximum(cost, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    
    # UCB exploration term
    # beta * sigma encourages exploring regions with high uncertainty
    # Use a moderate beta value
    beta = 2.0
    ucb = beta * sigma
    
    # Combined score: balance exploitation (EI) and exploration (UCB)
    # Normalize by cost to account for evaluation cost
    # Use a weighted combination
    score = (ei + ucb) / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(score), score, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
