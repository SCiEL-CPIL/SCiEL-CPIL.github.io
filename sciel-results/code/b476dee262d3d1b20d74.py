import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Clip sigma to avoid division by zero and numerical instability
    sigma = np.clip(sigma, 1e-9, None)
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (it should be, but numerical errors might make it tiny negative)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware penalty: use a smooth, bounded penalty that favors low-cost points
    # cost is in [1, 2], so (cost - 1) is in [0, 1]
    # Use a quadratic penalty to smooth the transition and avoid harsh cutoffs
    # lambda_penalty controls the strength of cost aversion
    lambda_penalty = 2.0
    penalty = 1.0 - lambda_penalty * (cost - 1.0) ** 2
    
    # Ensure penalty is non-negative
    penalty = np.maximum(penalty, 0.0)
    
    scores = ei * penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
