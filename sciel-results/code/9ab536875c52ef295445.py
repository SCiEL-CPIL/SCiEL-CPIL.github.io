import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Probability of Improvement (PI)
    # PI(x) = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # This strategy favors points where there is a high chance of improvement,
    # even if the expected magnitude of improvement is small.
    # It provides a distinct exploration bias compared to EI, which can be
    # overly greedy in the early stages or too conservative in the late stages.
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate PI score
    # norm.cdf(z) gives the probability that a normal distribution with mean 0 and std 1
    # is less than z. Since f(x) ~ N(mu, sigma^2), P(f(x) < f_best) is exactly this.
    score = norm.cdf(z)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
