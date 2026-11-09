import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Thompson Sampling: draw a random sample from the posterior for each candidate
    # sample = mu + sigma * z, where z ~ N(0,1)
    # This introduces stochastic exploration that can escape local minima
    # in the acquisition landscape that deterministic EI gets stuck in.
    
    # Generate random Gaussian perturbations
    z = np.random.standard_normal(len(mu))
    samples = mu + sigma * z
    
    # Cost-aware normalization: divide by cost to prefer cheaper evaluations
    # Use sqrt(cost) to soften the penalty, allowing some exploration of expensive regions
    safe_cost = np.maximum(cost, 1e-9)
    scores = samples / np.sqrt(safe_cost)
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
