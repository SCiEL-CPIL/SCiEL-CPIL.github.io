import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    eps = 1e-9
    
    # Expected Improvement
    z = (f_best - mu) / (sigma + eps)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Use a softer penalty for cost to balance exploitation and cost
    # The seed used 1/cost. Let's try 1/(cost^0.5) or similar
    # Or maybe just EI without cost penalty? But cost matters.
    # Let's try EI / (cost + 0.5) to avoid division by very small costs dominating
    # Or maybe the seed is actually better?
    # The seed scored 1.0. The current UCB scored 0.32.
    # Let's go back to EI but maybe with a slight modification.
    
    # The seed: ei / cost
    # Let's try that.
    scores = ei / (cost + eps)
    
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    return scores
