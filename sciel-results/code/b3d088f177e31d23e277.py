# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        # Fallback: return UCB-like scores quickly
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + 2.0 * sigma_safe

    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)

    # Predictive Entropy Search (PES) variant
    # Score each candidate by expected reduction in entropy of the global minimum
    # Approximated via mutual information between f(x) and the event that x is the new minimum
    
    # The key insight for PES: we want to find points where the posterior
    # distribution of f(x) is most informative about where the global minimum is.
    # 
    # A practical approximation: score = P(f(x) < f_best) * log(1 + sigma(x))
    # This balances:
    # 1. Probability of improvement (exploitation)
    # 2. Uncertainty (exploration) - high sigma means more information gain
    
    # Compute probability of improvement: P(f(x) < f_best)
    # f(x) ~ N(mu, sigma^2)
    z = (f_best - mu) / sigma_safe
    prob_improve = norm.cdf(z)
    
    # Information gain term: log(1 + sigma) captures uncertainty
    # Higher sigma means more potential information about the function's structure
    info_gain = np.log1p(sigma_safe)
    
    # Combined PES-like score
    # We weight both terms to balance exploration and exploitation
    score = prob_improve * (1.0 + 1.5 * info_gain)
    
    # Add a small UCB component to ensure we don't ignore high-uncertainty regions
    # where the probability of improvement might be low but information gain is high
    ucb_component = -mu + 1.0 * sigma_safe
    score = 0.7 * score + 0.3 * (ucb_component - np.min(ucb_component)) / (np.max(ucb_component) - np.min(ucb_component) + 1e-9)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
