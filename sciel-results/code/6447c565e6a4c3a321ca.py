import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-Adjusted Expected Improvement (CAEI) enhanced with a
    # Probability of Improvement (PI) weighting.
    # This models the "value" of an evaluation as the expected improvement
    # discounted by the confidence that the improvement is real, approximating
    # the probability of reaching the global minimum in a stochastic shortest
    # path framework.
    
    # Avoid division by zero or negative costs
    cost = np.maximum(cost, 1e-8)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Calculate Probability of Improvement (PI)
    pi = norm.cdf(z)
    
    # Calculate Expected Improvement (EI)
    ei = (f_best - mu) * pi + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Weight EI by PI to suppress exploration in high-uncertainty regions
    # where the probability of actual improvement is low.
    weighted_ei = ei * pi
    
    # The acquisition score is the weighted EI divided by cost
    scores = weighted_ei / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
