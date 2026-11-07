import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive for division
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Hybrid Score: EI + beta * sigma
    # beta controls exploration. 
    # For Rastrigin, a moderate exploration is needed to jump between minima.
    beta = 1.0
    
    # The term beta * sigma acts as an UCB bonus for minimization (since we want low values, 
    # high uncertainty is good). 
    # Note: Standard UCB for minimization is mu - beta*sigma. 
    # Maximizing (beta*sigma - mu) is equivalent to minimizing (mu - beta*sigma).
    # EI is roughly (f_best - mu) when f_best >> mu.
    # So EI + beta*sigma is roughly (f_best - mu) + beta*sigma.
    # This encourages low mu and high sigma.
    
    hybrid_score = ei + beta * sigma_safe
    
    # Normalize by cost
    scores = hybrid_score / cost
    
    # Replace non-finite values with 0
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
