import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure inputs are arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Avoid division by zero in z
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized distance
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration bonus: encourage high uncertainty
    # A value of 1.0 for beta is a standard starting point for UCB
    # We add it to EI to create a hybrid score
    beta = 1.0
    exploration_bonus = beta * sigma_safe
    
    # Combine EI and Exploration
    # This is effectively a UCB-like score but grounded in EI
    score = ei + exploration_bonus
    
    # Cost-awareness: divide by cost
    # Cost is >= 1
    score = score / cost
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
