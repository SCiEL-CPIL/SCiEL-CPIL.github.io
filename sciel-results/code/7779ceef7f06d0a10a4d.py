"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 280:
        # Fallback to simple EI/cost to ensure fast return
        z = (f_best - mu) / (sigma + 1e-9)
        z = np.clip(z, -30, 30)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / cost

    # Thompson Sampling: Sample from the posterior distribution of the objective
    # f(x) ~ N(mu, sigma^2)
    # We sample one value for each candidate point.
    # To ensure reproducibility within the same call (if needed, though not strictly 
    # required by the contract, it's good practice to use a fixed seed or rely on 
    # the randomness provided by the environment. Since the contract doesn't pass 
    # a seed, we use the global numpy random state).
    
    # Generate random samples
    # sigma is standard deviation
    samples = mu + sigma * np.random.randn(2000)
    
    # Calculate improvement for the sampled values
    # If the sampled value is better (lower) than f_best, we have an improvement.
    # Improvement = f_best - samples
    improvement = f_best - samples
    
    # Cost-aware score
    # We want to maximize the expected improvement per unit cost.
    # In TS, we treat the sample as the "true" value and pick the one with highest 
    # improvement/cost.
    # To avoid division by zero or very small costs (cost is 1 + mean(x), min 1),
    # cost is always >= 1.
    
    score = improvement / cost
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
