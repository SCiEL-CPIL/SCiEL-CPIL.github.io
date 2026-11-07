"""Cost-aware acquisition for BO on three hump camel 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: total wall-clock budget is ~330s.
# Track elapsed time to ensure we don't exceed the limit.
_start_time = time.time()
TIME_BUDGET = 320.0  # Safety margin before 330s

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a structural change to Thompson Sampling
    with cost-awareness. This is a different design family from EI/UCB.
    
    Parameters:
        mu: (2000,) GP mean predictions
        sigma: (2000,) GP std dev predictions (positive)
        f_best: float, best observed value so far
        cost: (2000,) evaluation cost per candidate
    
    Returns:
        (2000,) finite acquisition scores, higher = better
    """
    # Deadline guard: if we're getting close to the time limit, return a simple
    # deterministic score to avoid timeout
    elapsed = time.time() - _start_time
    if elapsed > TIME_BUDGET:
        # Return a simple cost-based score as fallback
        return 1.0 / np.maximum(cost, 1.0)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Thompson Sampling: sample from posterior and pick the best
    # For minimization, we sample f(x) ~ N(mu, sigma^2) and score = -sample
    # This is a structural change from EI/UCB to a sampling-based approach
    
    # Generate random samples from the posterior
    # Use a single sample per call (this is the standard Thompson Sampling approach)
    # The randomness is handled by the GP's uncertainty in sigma
    rng = np.random.default_rng()
    samples = mu + sigma_safe * rng.standard_normal(len(mu))
    
    # Score is negative of the sample (since we want to minimize)
    # Higher score = lower predicted value = better
    raw_score = -samples
    
    # Cost-awareness: divide by cost to prefer cheaper evaluations
    # cost is in [1, 2], so 1/cost is in [0.5, 1.0]
    cost_factor = 1.0 / np.maximum(cost, 1.0)
    
    score = raw_score * cost_factor
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
