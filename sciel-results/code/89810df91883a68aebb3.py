"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global start time for deadline guard
_START_TIME = time.time()
DEADLINE = 330.0
SAFETY_MARGIN = 10.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Strategy:
    - Sample from the posterior predictive distribution for each candidate.
    - Score candidates by the negative sampled value divided by the evaluation cost.
    - This encourages exploration (via sampling) and exploits cost-efficiency.
    """
    # Deadline check
    elapsed = time.time() - _START_TIME
    if elapsed > DEADLINE - SAFETY_MARGIN:
        # Fallback: Return a safe, finite score. 
        # If we are out of time, we can't do complex sampling reliably in the loop context 
        # if this function is called repeatedly in a tight loop without reset, but 
        # the contract implies the function is called per step. 
        # To be safe, return a simple EI-based score which is fast.
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)

    # Ensure sigma is positive for sampling
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Generate random samples from the posterior predictive distribution
    # mu and sigma are in objective units.
    # We want to minimize f, so we sample f_tilde ~ N(mu, sigma^2)
    # and we want to pick the smallest f_tilde.
    # The evaluator picks the candidate with the HIGHEST score.
    # So we define score = -f_tilde / cost.
    
    # Vectorized sampling
    samples = np.random.normal(loc=mu, scale=sigma_safe)
    
    # Calculate cost-aware score
    # Higher score = better candidate to evaluate next
    # We want low function values and low cost.
    # score = -sampled_value / cost
    scores = -samples / (cost + 1e-9)
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
