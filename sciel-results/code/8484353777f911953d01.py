import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Hybrid acquisition: blend exploitation (-mu) with exploration (UCB-like)
    # For 1D sphere with 30 evals, we need to balance finding the minimum near 0
    # with exploring uncertain regions to avoid getting stuck.
    
    sigma_safe = np.where(sigma > 1e-12, sigma, 1e-12)
    
    # Expected Improvement for minimization
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB with adaptive beta: start with more exploration, reduce over time
    # Since we don't know the step, use f_best magnitude to adapt
    # When f_best is large, we're far from optimum -> explore more
    # When f_best is small, we're close -> exploit more
    beta = 2.0 * np.sqrt(np.log1p(1.0 / np.maximum(np.abs(f_best), 1e-6)))
    beta = np.clip(beta, 0.5, 4.0)
    
    # Blend EI and UCB
    ucb = -mu + beta * sigma_safe
    
    # Use a weighted combination: weight EI more when close to optimum
    weight_ei = 1.0 / (1.0 + np.abs(f_best))
    scores = weight_ei * ei + (1 - weight_ei) * ucb
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    return scores
