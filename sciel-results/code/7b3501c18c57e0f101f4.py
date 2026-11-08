import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    
    # Initialize timing if not set
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: 330s wall-clock limit, leave 5s margin
    if time.time() - _start_time > 325:
        mu = np.asarray(mu)
        sigma = np.asarray(sigma)
        cost = np.asarray(cost)
        # Simple fallback: Cost-Adjusted UCB
        scores = (-mu + 2.0 * sigma) / (cost + 1e-8)
        return np.where(np.isfinite(scores), scores, -np.inf)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is positive and finite
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Small epsilon to prevent division by zero
    eps = 1e-8
    safe_sigma = np.maximum(sigma, eps)
    
    # Calculate z-score
    z = (f_best - mu) / safe_sigma
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Exploration bonus: lambda * sigma
    # This encourages exploration in high uncertainty regions, penalized by cost
    lambda_explore = 0.5
    
    # Cost-Adjusted Score
    # We want high scores for high EI and high sigma (uncertainty), low cost
    scores = (ei + lambda_explore * sigma) / cost
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
