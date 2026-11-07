"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Knowledge Gradient (KG)-inspired acquisition function for cost-aware Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best (lowest) observed value so far
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate (1 + mean of scaled coordinates)
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    start_time = time.time()
    
    # Ensure inputs are arrays with correct shape
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    
    # Handle edge cases
    if n == 0:
        return np.array([])
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute z-scores for expected improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scaling: use 1/sqrt(cost) as a sublinear discount
    # This balances exploration of high-cost regions against budget efficiency
    cost_discount = 1.0 / np.sqrt(np.maximum(cost, 1e-10))
    
    # KG-inspired score: combine EI with uncertainty and cost awareness
    # The idea is to estimate the expected improvement in the global optimum
    # by weighting EI by the probability that this point could be the new minimum
    # We approximate this by using EI itself (which already captures this) 
    # and scaling by cost discount
    
    # Add a small exploration bonus based on uncertainty relative to cost
    # This helps explore high-uncertainty, low-cost regions
    exploration_bonus = sigma_safe * cost_discount * 0.1
    
    # Final score: EI per unit cost (sublinear) plus exploration bonus
    scores = ei * cost_discount + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard: if we're taking too long, return what we have
    elapsed = time.time() - start_time
    if elapsed > 5.0:  # Safety margin well within 300s limit
        return scores
    
    return scores
# EVOLVE-BLOCK-END
