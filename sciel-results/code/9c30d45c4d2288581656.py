# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Budget-aware acquisition function using Expected Improvement per unit cost.
    We compute EI and divide by cost to get a cost-normalized score.
    We add a small exploration term based on sigma to encourage diversity early.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero in z
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Clip z to avoid numerical issues with norm.cdf/pdf for extreme values
    z_clipped = np.clip(z, -10, 10)
    
    ei = improvement * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative (numerical noise can make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Cost-normalized score
    # We want to maximize EI / cost
    # But we also want to balance exploration. 
    # A simple approach: score = EI / cost
    # However, if cost is very low (early on), this might be too aggressive.
    # Let's stick to the standard cost-aware EI.
    
    # Add a small exploration bonus to avoid getting stuck in local optima
    # The bonus is proportional to sigma, scaled by a factor that decreases as we learn
    # But we don't have a direct measure of "how much we've learned" other than cost.
    # Let's use a simple heuristic: exploration bonus = sigma * 0.1
    # This encourages exploring areas with high uncertainty.
    
    # Actually, the standard cost-aware EI is often sufficient.
    # Let's try a slightly more robust version:
    # score = EI / (cost + epsilon)
    # where epsilon is a small constant to avoid division by zero.
    
    score = ei / (cost + 1e-9)
    
    # Ensure scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
