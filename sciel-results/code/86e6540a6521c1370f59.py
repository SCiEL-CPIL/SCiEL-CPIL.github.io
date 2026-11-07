import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (POI)
    normalized by cost, with an exploration bonus.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma and cost are positive
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Probability of Improvement (POI)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    poi = norm.cdf(z)
    
    # Expected Improvement (EI) for reference, but we use POI as primary
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # phi(z) = norm.pdf(z)
    # ei = (f_best - mu) * poi + sigma * norm.pdf(z)
    
    # Use POI as the main acquisition signal
    # Add a small exploration term to avoid getting stuck
    # The exploration term rewards high uncertainty (sigma)
    # We can blend POI and an exploration term
    
    # Cost-aware: divide by cost
    # To avoid division by zero or very small costs, we already clamped cost
    
    # Simple cost-aware POI
    scores = poi / cost
    
    # Add an exploration bonus that is also cost-aware
    # The bonus should be small but enough to encourage exploration
    # A common approach is to add beta * sigma / cost
    beta = 0.5
    exploration = beta * sigma / cost
    
    scores = scores + exploration
    
    # Handle non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
