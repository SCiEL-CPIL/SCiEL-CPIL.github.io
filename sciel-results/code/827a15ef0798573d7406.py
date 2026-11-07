"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Samples from the GP posterior and selects points based on the 
    ratio of negative sampled value to cost. This naturally balances 
    exploration (high sigma leads to high variance in samples) and 
    exploitation (low mu leads to low sampled values).
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 1e-12), cost, 1.0)
    
    sigma = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling: Sample from the posterior
    # We use a fixed seed for reproducibility in the sample generation? 
    # No, the task implies stochastic acquisition is allowed as long as output is finite.
    # However, using a random sample every call is standard for TS.
    # To avoid issues with global RNG state affecting other parts, we use a local random generator.
    rng = np.random.default_rng()
    
    # Sample function values
    # We want to minimize f, so we want low sampled values.
    # Score = -sampled_value / cost
    samples = rng.normal(mu, sigma)
    
    # Calculate scores
    # Higher score is better.
    # We want low function values, so we negate the sample.
    # We divide by cost to account for the expense.
    scores = -samples / cost
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # If all scores are -inf (unlikely), return zeros
    if not np.any(np.isfinite(scores)):
        return np.zeros_like(mu)
        
    return scores
# EVOLVE-BLOCK-END
