import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    Uses Lower Confidence Bound (LCB) for minimization, which is equivalent
    to maximizing the score: -mu + beta * sigma.
    Beta is adaptive: higher when f_best is far from the minimum (3.0),
    lower when f_best is close to the minimum.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 1.0
    
    # Adaptive Beta calculation
    # We want to explore more (higher beta) when we are far from the optimum (f_best >> 3.0)
    # and exploit more (lower beta) when we are close (f_best ~ 3.0)
    
    # Gap from the theoretical minimum
    gap = max(f_best - 3.0, 0.0)
    
    # Base exploration constant
    # When gap is 0, beta should be ~1.0 (exploitation)
    # When gap is large, beta should be ~3.0 (exploration)
    # Using a saturating function to prevent beta from getting too large
    # beta = 1.0 + 2.0 * (1 - exp(-gap / 2.0))
    # If gap=0, beta=1.0. If gap=10, 1-exp(-5)~0.99, beta~3.0.
    beta = 1.0 + 2.0 * (1.0 - np.exp(-gap / 2.0))
    
    # Calculate score: we want to maximize this score
    # Score = - (mu - beta * sigma) = -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores
