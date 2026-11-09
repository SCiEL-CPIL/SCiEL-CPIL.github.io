"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track progress if needed, though TS is stateless per call
# We can use a global seed or just rely on np.random
# To ensure reproducibility or specific behavior, we might want to control the random state
# but the evaluator likely resets or runs in a fresh process? 
# "For each seed 0, 1 and 2 the evaluator starts from 5 uniform random points..."
# This implies the Python process might persist or restart. 
# If it persists, np.random state continues. That's fine.
# If it restarts, that's also fine.

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Independent Thompson Sampling.
    
    Instead of computing an expected improvement (which can saturate) or a fixed
    UCB (which can be too deterministic), we sample from the posterior predictive
    distribution for each candidate point.
    
    For minimization, we want to find the point with the lowest predicted value.
    We simulate the objective value y at each point:
    y ~ N(mu, sigma^2)
    
    We return scores such that higher scores correspond to lower predicted y.
    score = -y = -(mu - sigma * Z) = -mu + sigma * Z
    where Z is a standard normal random variable.
    
    This naturally balances exploration (high sigma leads to high variance in y, 
    potentially leading to very low y if Z is large positive) and exploitation 
    (low mu leads to low y).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive
    sigma = np.maximum(sigma, 1e-12)
    
    # Draw random samples from standard normal distribution
    # Shape matches mu and sigma
    Z = np.random.randn(*mu.shape)
    
    # Compute Thompson Sampling scores
    # We want to minimize y, so we maximize -y
    # y = mu - sigma * Z  (Note: standard normal Z has mean 0, var 1. 
    # If we define y = mu + sigma * Z, it's also valid. The sign of Z doesn't matter 
    # for the distribution, but it affects the specific sample. 
    # Let's use y = mu + sigma * Z. Then score = -y = -mu - sigma * Z.
    # Or y = mu - sigma * Z. Then score = -mu + sigma * Z.
    # Both are equivalent in distribution.
    
    scores = -mu + sigma * Z
    
    # Ensure finite outputs
    # If sigma is huge and Z is huge, this could be inf, but sigma is from GP, usually bounded.
    # Z is from randn, usually bounded in practice (rarely > 6).
    # mu is from GP, usually bounded.
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
