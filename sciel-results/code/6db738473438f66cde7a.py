import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: avoid division by zero in z calculation
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in norm.cdf/pdf for extreme values
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Hybrid with UCB-like exploration term
    # Exploration term: sigma (uncertainty)
    # We add a weighted sigma to encourage exploration
    # beta = 1.0 is a reasonable starting point for a 1:1 weight between EI and sigma
    beta = 1.0
    exploration_term = beta * sigma
    
    # Combined numerator: Exploitation + Exploration
    numerator = ei + exploration_term
    
    # Cost-aware penalty: use sqrt(cost) as suggested by insights
    # cost is provided, ensure it's positive
    cost_safe = np.maximum(cost, 1e-6)
    penalty = np.sqrt(cost_safe)
    
    # Final score
    scores = numerator / penalty
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
