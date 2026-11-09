# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a dynamic UCB weight that scales with the median sigma,
    approximating the ratio of remaining uncertainty (budget) to current scale.
    This shifts from exploration (high beta when sigma is high) to exploitation
    (low beta when sigma is low) as the GP converges.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x scaled to [0,1])).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Dynamic Beta Calculation
    # Use median sigma as a proxy for "remaining budget" (uncertainty).
    # Early in optimization, sigma is high -> high beta (exploration).
    # Late in optimization, sigma is low -> low beta (exploitation).
    median_sigma = np.median(sigma)
    
    # Scale beta. 
    # If median_sigma is ~1.0 (standardized initial), beta is ~1.5.
    # If median_sigma is ~0.1, beta is ~0.15.
    beta = 1.5 * median_sigma
    
    # Clip beta to prevent extreme values
    beta = np.clip(beta, 0.1, 2.0)
    
    # Calculate Score
    # We maximize (EI + beta * sigma) / cost
    # EI handles the improvement expectation.
    # beta * sigma adds an exploration bonus scaled by dynamic beta.
    # Dividing by cost prioritizes cheaper evaluations.
    
    scores = (ei + beta * sigma) / cost
    
    # Handle any non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
