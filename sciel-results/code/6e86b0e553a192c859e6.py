"""Rewritten program: Quantile-Based Cost-Discounted Lower Confidence Bound."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Quantile-Based Cost-Discounted Lower Confidence Bound.
    
    Maps the posterior predictive distribution to the 5th percentile (aggressive
    lower tail) and applies a non-linear sigmoidal cost discount to aggressively
    prune high-cost candidates in the outer regions of the domain.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Mean of the GP posterior predictive distribution.
    sigma : array of shape (2000,)
        Standard deviation of the GP posterior predictive distribution.
    f_best : float
        Best observed function value so far.
    cost : array of shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = better).
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero in sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Use the 5th percentile for a robust but aggressive lower bound
    # z_0.05 ≈ -1.6449
    quantile_z = norm.ppf(0.05)
    
    # Compute the quantile-based value (lower is better for the objective)
    quantile_value = mu + quantile_z * sigma_safe
    
    # We want to minimize, so lower quantile_value is better.
    # Convert to a "score" where higher is better: negative of quantile_value.
    # This is equivalent to a Lower Confidence Bound (LCB) score.
    base_score = -quantile_value
    
    # Apply non-linear sigmoidal cost discount.
    # Cost ranges from 1.0 to 2.0.
    # We want to heavily discount high costs (near 2.0) and preserve low costs (near 1.0).
    # Sigmoid center at 1.6, steepness k=8.
    # D(c) = 1 / (1 + exp(k * (c - c_center)))
    # At c=1.0: D ≈ 0.999
    # At c=1.6: D = 0.5
    # At c=2.0: D ≈ 0.029
    # This aggressively prunes expensive points.
    c_center = 1.6
    k = 8.0
    discount = 1.0 / (1.0 + np.exp(k * (cost - c_center)))
    
    # Combine base score with cost discount.
    # Multiplying the positive base score by the discount reduces the score
    # for expensive candidates.
    scores = base_score * discount
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
