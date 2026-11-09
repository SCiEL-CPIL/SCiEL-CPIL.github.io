import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of 1D sphere on [-5, 5].
    
    Uses a hybrid approach: Expected Improvement (EI) with a small exploration bonus
    based on sigma, to balance exploitation and exploration.
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement: how many std devs below f_best is the mean
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Add a small exploration bonus to avoid getting stuck in local minima
    # The bonus is proportional to sigma, encouraging exploration in uncertain regions
    # but scaled down so that EI dominates when there's clear improvement
    exploration_weight = 0.1
    exploration_bonus = exploration_weight * sigma_safe
    
    return ei + exploration_bonus
