"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Compute cost-aware acquisition scores for 2000 candidate points.
    
    Uses a hybrid approach combining:
    1. Expected Improvement (EI) per unit cost
    2. Upper Confidence Bound (UCB) per unit cost
    3. Weighted by confidence in the prediction (low sigma regions get more exploitation weight)
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Mean of GP predictions
    sigma : array of shape (2000,)
        Standard deviation of GP predictions
    f_best : float
        Best observed function value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = more promising to evaluate)
    """
    # Safety: ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace any non-finite values with safe defaults
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-8, 1e6)
    
    # Standardize for stability
    z = (f_best - mu) / sigma
    
    # Clip z to avoid extreme values in normal CDF/PDF
    z = np.clip(z, -10, 10)
    
    # Compute Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (it should be, but numerical issues can cause small negatives)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB with adaptive beta
    # Use a moderate beta that balances exploration and exploitation
    beta = 0.8
    ucb = mu + beta * sigma
    
    # For minimization, we want lower values, so UCB for minimization is:
    # We want to find points where the lower bound is promising
    # Actually, for minimization, we should use: mu - beta * sigma (lower confidence bound)
    lcb = mu - beta * sigma
    
    # Cost-normalized scores
    ei_per_cost = ei / cost
    lcb_per_cost = -lcb / cost  # Negate because we want to minimize
    
    # Confidence weight: points with lower uncertainty are more reliable
    # Use a smooth weighting that favors exploitation when sigma is low
    # sigma_max is the maximum sigma for normalization
    sigma_max = np.max(sigma)
    if sigma_max > 0:
        sigma_norm = sigma / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma)
    
    # Weight: higher when sigma is low (more confident predictions)
    # Use exponential decay: w = exp(-alpha * sigma_norm)
    alpha = 1.0
    weight = np.exp(-alpha * sigma_norm)
    
    # Blend EI and LCB based on confidence
    # When confident (low sigma), rely more on LCB (exploitation)
    # When uncertain (high sigma), rely more on EI (balanced)
    # Actually, let's think about this differently:
    # EI already balances exploration and exploitation
    # LCB is more exploitative
    # So we can blend: score = weight * lcb_per_cost + (1 - weight) * ei_per_cost
    
    scores = weight * lcb_per_cost + (1.0 - weight) * ei_per_cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Return the scores (higher = more promising)
    return scores
# EVOLVE-BLOCK-END
