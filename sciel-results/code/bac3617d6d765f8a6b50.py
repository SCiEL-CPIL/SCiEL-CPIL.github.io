"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Probabilistic Regret Lower Bound (PRLB)
    # Balances exploitation (expected improvement) and exploration (uncertainty)
    # using the probability of improvement as a weight.
    
    # Avoid division by zero
    sigma_safe = sigma + 1e-9
    
    # Z-score for Probability of Improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability that f(x) < f_best
    p = norm.cdf(z)
    
    # Exploitation term: Weighted expected improvement
    # When p is high (mu << f_best), this approximates (f_best - mu)
    exploit_term = p * (f_best - mu)
    
    # Exploration term: Weighted uncertainty
    # When p is low (mu >> f_best or high sigma), this approximates sigma
    explore_term = (1.0 - p) * sigma_safe
    
    # Cost-normalized score
    # Higher score is better
    score = (exploit_term + explore_term) / cost
    
    # Ensure output is finite and of correct shape
    # norm.cdf returns finite values for finite inputs
    # cost is positive
    return score
# EVOLVE-BLOCK-END
