import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    # Safety: ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (for minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z))
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive exploration bonus:
    # Scale the sigma term by how uncertain the GP is relative to the spread of predictions.
    # This prevents over-exploration when the GP is already confident.
    mu_range = np.max(mu) - np.min(mu)
    if mu_range < 1e-12:
        mu_range = 1e-12
    
    # Relative uncertainty: how much does sigma vary compared to the mu spread?
    sigma_mean = np.mean(sigma_safe)
    if sigma_mean < 1e-12:
        sigma_mean = 1e-12
    
    # Exploration coefficient: moderate, scaled by relative uncertainty
    # When sigma is large relative to mu spread, explore more
    beta = 0.5 * (sigma_mean / mu_range)
    beta = np.clip(beta, 0.1, 2.0)
    
    # Combined score: EI + beta * sigma, all divided by cost
    # This is a single, smooth, cost-aware acquisition
    score = (ei + beta * sigma_safe) / cost_safe
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # If all scores are the same, return a small random perturbation to break ties
    if np.max(score) - np.min(score) < 1e-15:
        score = score + np.random.randn(len(score)) * 1e-10
    
    return score
