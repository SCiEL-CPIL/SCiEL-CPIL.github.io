"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_DEADLINE = time.time() + 290.0  # 300s limit, keep 10s margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Strategy:
    1. Sample a function value y_sample from N(mu, sigma) for each candidate.
    2. The "score" is the negative of this sampled value, normalized by cost.
    3. This encourages selecting points where the sampled function is low (exploit)
       or where uncertainty is high and the sample happens to be low (explore).
       Specifically, Thompson Sampling picks the argmin of the sample. 
       To fit the "higher score = evaluate next" interface, we return -y_sample / cost.
       
    Why this is different:
    - EI/LCB are deterministic functions of (mu, sigma).
    - Thompson Sampling introduces stochasticity. In a loop, this allows the algorithm
      to "try" different regions repeatedly if their samples look promising, 
      which can escape local optima in Rastrigin more effectively than a greedy
      deterministic bound.
    
    Note on Cost:
    The cost is 1 + mean(x_scaled). Lower cost is better.
    We divide the benefit (improvement over current best or just low value) by cost.
    For Thompson Sampling, the "benefit" is the value of the sample relative to 0 (global min)
    or relative to f_best.
    Standard TS minimizes f(x). So we want to maximize -f_sample(x).
    Score = -f_sample(x) / cost(x).
    """
    
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback to a simple deterministic score if we are running out of time
        # This shouldn't happen often, but prevents timeout failure
        return -mu / cost

    # 1. Sample from the posterior
    # sigma is standard deviation
    # Use standard normal random numbers
    z = np.random.normal(0, 1, size=mu.shape)
    y_sample = mu + sigma * z
    
    # 2. Calculate Score
    # We want to minimize f(x). 
    # If y_sample is low, it's a good candidate.
    # Score should be high for low y_sample.
    # So Score = -y_sample.
    
    # Cost-aware: Divide by cost.
    # cost is (2000,) array.
    # Ensure cost is not zero (it's 1 + mean(x), so min is 1).
    
    score = -y_sample / cost
    
    # Handle non-finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
