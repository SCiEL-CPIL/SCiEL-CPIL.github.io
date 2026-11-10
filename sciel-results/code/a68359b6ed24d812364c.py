import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-Adjusted Expected Improvement (CAEI) with Non-Linear Scaling
    # This approximates a value function where the marginal utility of improvement
    # increases with the magnitude of the improvement, effectively prioritizing
    # points with high expected improvement more strongly (exploitation boost).
    
    # Avoid division by zero or negative costs
    cost = np.maximum(cost, 1e-8)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma
    
    # Standard formula
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Non-linear score: EI^2 / (cost * (EI + eps))
    # This behaves like EI/cost for large EI, but penalizes small EI more heavily,
    # mimicking a dynamic programming value function that favors significant reductions
    # in the cost-to-go.
    eps = 1e-6
    scores = (ei * ei) / (cost * (ei + eps))
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
