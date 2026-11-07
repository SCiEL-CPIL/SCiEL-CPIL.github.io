"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track time for deadline guard
_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB).
    
    Args:
        mu: Mean of the GP posterior (shape 2000,)
        sigma: Standard deviation of the GP posterior (shape 2000,)
        f_best: Current best observed value
        cost: Cost of evaluating each candidate (shape 2000,)
        
    Returns:
        scores: Acquisition scores (shape 2000,), higher is better.
    """
    current_time = time.time()
    elapsed = current_time - _start_time
    
    # Deadline guard: if we are close to the 330s limit, use a fast fallback
    # to ensure the function returns in time.
    if elapsed > 320.0:
        # Fallback: Greedy minimization adjusted for cost
        # Avoid division by zero (though cost >= 1)
        return -mu / cost

    # UCB for minimization:
    # We want to maximize the "optimistic" value.
    # For minimization, the optimistic value is mu - beta * sigma.
    # To maximize this, we use -mu + beta * sigma.
    # Dividing by cost makes it cost-aware.
    beta = 1.5
    score = (-mu + beta * sigma) / cost
    
    return score
# EVOLVE-BLOCK-END
