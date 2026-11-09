import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Sanitize inputs
    sigma = np.asarray(sigma, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    sigma = np.maximum(sigma, 1e-8)
    sigma = np.nan_to_num(sigma, nan=1e-8, posinf=1e6, neginf=1e-8)
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    cost = np.maximum(cost, 1e-8)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1.0)

    # Thompson Sampling: sample from GP posterior for each candidate
    # Sample: y_i = mu_i - sigma_i * z_i where z_i ~ N(0,1)
    # We want to minimize, so lower sampled values are better.
    # Score should be higher for candidates that are likely to improve f_best.
    
    # Generate random samples for Thompson Sampling
    rng = np.random.default_rng()
    n = mu.shape[0]
    z_samples = rng.standard_normal(n)
    
    # Sampled function values (lower is better for minimization)
    sampled_values = mu - sigma * z_samples
    
    # Score: how much improvement we expect relative to f_best, normalized by cost
    # Higher score = more likely to be a good next point
    # We want points where sampled_value < f_best, so improvement = f_best - sampled_value
    # But we also need to account for cost
    
    # Expected improvement based on the sample
    improvement = np.maximum(0, f_best - sampled_values)
    
    # Cost-aware score: improvement per unit cost
    scores = improvement / cost
    
    # Add a small exploration bonus for high uncertainty regions
    # This helps escape local optima in the multi-modal Shekel landscape
    exploration_bonus = 0.02 * sigma / cost
    
    scores = scores + exploration_bonus
    
    # Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

    return scores
