"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function that balances expected improvement against cost.
    
    Uses a cost-aware variant of Expected Improvement with an additional
    exploration term to encourage sampling in uncertain regions.
    
    Parameters:
    -----------
    mu : array of shape (N,)
        Mean predictions from GP
    sigma : array of shape (N,)
        Standard deviation predictions from GP
    f_best : float
        Best observed value so far (lower is better)
    cost : array of shape (N,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (N,)
        Acquisition scores (higher = more promising)
    """
    # Safety: ensure we don't spend too long
    t0 = time.time()
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-12)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Use stable computation for normal CDF and PDF
    # For large negative z, Phi(z) is very small
    # For large positive z, Phi(z) is close to 1
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware scaling: divide by cost to get improvement per unit cost
    # Also add a small exploration bonus based on uncertainty
    # This helps explore regions with high uncertainty even if EI is low
    
    # Normalize cost for better conditioning
    cost_safe = np.maximum(cost, 1e-6)
    
    # Cost-aware EI
    scores = ei / cost_safe
    
    # Add exploration term: reward high uncertainty in cheap regions
    # This is a UCB-like term that encourages exploration
    # Scale it to be comparable to EI
    exploration_weight = 0.1
    exploration = exploration_weight * sigma / cost_safe
    
    scores = scores + exploration
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time
    if time.time() - t0 > 280:
        # Return what we have with safety margin
        return scores
    
    return scores
# EVOLVE-BLOCK-END
