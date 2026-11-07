import numpy as np
import time

# Global start time for deadline guard, though likely not needed for 2000 points
_start_time = time.time()
DEADLINE = 330.0

def acquisition(mu, sigma, f_best):
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Ensure sigma is non-negative to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Deadline guard
    # If we are close to the deadline, return a deterministic score that prioritizes
    # the current best estimate to ensure a valid return quickly.
    # In practice, this function is O(N) and very fast.
    if time.time() > DEADLINE - 5.0:
        # Return scores based purely on exploitation (low mu)
        # to be safe and fast
        return -mu

    # UCB for minimization:
    # We want to maximize the score.
    # The Lower Confidence Bound (LCB) is mu - kappa * sigma.
    # Minimizing LCB is equivalent to maximizing -LCB = -mu + kappa * sigma.
    
    # Kappa controls the exploration-exploitation trade-off.
    # Higher kappa means more exploration (higher weight on sigma).
    # For Weierstrass (multimodal), a higher kappa is often beneficial
    # to escape local minima.
    kappa = 3.0
    
    scores = -mu + kappa * sigma
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
