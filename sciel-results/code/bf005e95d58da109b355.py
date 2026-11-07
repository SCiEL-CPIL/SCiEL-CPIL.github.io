"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB)
    normalized by cost, with a decaying exploration parameter.
    """
    start_time = time.time()
    
    # Clamp sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-9, None)
    
    # Determine exploration parameter kappa
    # Use cost as a proxy for time/iterations
    # Start with high exploration, decay it
    # cost ranges from ~5 to 30
    # Let's use a square root decay as hinted by inspirations
    # Base kappa of 1.5 seems reasonable for 2D problems
    kappa = 1.5 / np.sqrt(cost / 5.0)
    
    # UCB for minimization: mu - kappa * sigma
    # We want to maximize the acquisition score, so we negate the UCB
    # and divide by cost to prioritize efficiency
    ucb = mu - kappa * sigma
    score = -ucb / cost
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=0.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 10.0:
        return score

    return score
# EVOLVE-BLOCK-END
