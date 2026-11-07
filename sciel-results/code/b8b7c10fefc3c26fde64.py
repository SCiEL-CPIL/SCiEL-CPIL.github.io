import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate EI
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Calculate UCB (Minimization: mu - beta * sigma)
    # Higher score is better for selection, so we want to maximize (mu - beta*sigma)??
    # No, for minimization, UCB is usually defined as:
    # Score = mu - beta * sigma
    # We want to pick the candidate with the LOWEST mu - beta*sigma?
    # Wait, the task says "Return 2000 finite scores, higher meaning evaluate next".
    # So if we use UCB for minimization:
    # We want to find points where f(x) is likely to be low.
    # f(x) ~ N(mu, sigma).
    # Lower bound of confidence interval: mu - beta*sigma.
    # We want to maximize the "potential" for being low?
    # No, we want to pick the point that minimizes the expected value or has high uncertainty.
    # Standard UCB for minimization:
    # Select x that minimizes: mu(x) - beta * sigma(x)
    # But we must return a SCORE where HIGHER is better.
    # So Score = -(mu - beta * sigma) = beta * sigma - mu.
    
    ucb_score = 5.0 * sigma - mu
    
    # Threshold for switching
    # Weierstrass on [-0.5, 0.5]
    # If f_best is high, explore.
    # If f_best is low, exploit.
    
    threshold = 0.1
    
    if f_best > threshold:
        scores = ucb_score
    else:
        scores = ei
        
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
