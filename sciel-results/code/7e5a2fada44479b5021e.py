"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Structural Change: Switch from Upper Confidence Bound (UCB) to Expected Improvement (EI).
    # EI is an integral-based probabilistic criterion that directly maximizes the expected reduction
    # in the objective function. It naturally balances exploration and exploitation based on the
    # posterior uncertainty and the current best value, without requiring a fixed beta parameter.
    
    start_time = time.time()
    
    # Handle zero or near-zero sigma to avoid division by zero
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate the z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    # If mu < f_best, z is positive (improvement likely).
    # If mu > f_best, z is negative (improvement unlikely, but sigma might make it possible).
    
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is CDF and phi is PDF of the standard normal distribution.
    
    # Optimization: Use vectorized scipy operations
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    scores = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Deadline Guard
    # Check if we are close to the time limit (300s). 
    # If so, return a safe fallback (greedy -mu) to ensure completion.
    if time.time() - start_time > 290:
        scores = -mu
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

    return scores
# EVOLVE-BLOCK-END
