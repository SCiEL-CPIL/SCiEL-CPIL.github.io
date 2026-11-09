import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration bonus (Uncertainty)
    # A common choice is to add a multiple of sigma
    # Let's try a blend that favors exploration in the early stages
    # Since we don't know the step count, we use a fixed weight.
    # The "Knowledge" suggests EI+UCB blend.
    # UCB (minimization) = mu - kappa * sigma.
    # We want to maximize score, so we use -UCB = kappa * sigma - mu.
    # Let's combine EI and -UCB.
    # Score = (w_ei * EI + w_ucb * (kappa * sigma - mu)) / cost
    
    # Let's start with a simple exploration bonus added to EI
    # This is equivalent to EI + lambda * sigma
    # Let's try lambda = 0.5
    score = (ei + 0.5 * sigma_safe) / cost
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
