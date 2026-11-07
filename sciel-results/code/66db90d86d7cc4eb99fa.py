import numpy as np
import time

def acquisition(mu, sigma, f_best):
    # Clip sigma to avoid division by zero and numerical instability
    sigma = np.maximum(sigma, 1e-8)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    # Strategy: Deterministic UCB for minimization
    # For minimization, we want to find regions where the function is likely to be low.
    # The posterior mean mu estimates the function value.
    # The posterior std sigma estimates the uncertainty.
    # A point is promising if:
    # 1. mu is low (exploit)
    # 2. sigma is high (explore, in case the true value is much lower than mu)
    
    # Upper Confidence Bound for minimization:
    # We want to maximize the probability of finding a low value.
    # This is equivalent to minimizing the "lower bound" of the function?
    # No, for minimization, the "optimistic" estimate is mu - kappa * sigma.
    # We pick the point with the lowest (mu - kappa * sigma).
    # Therefore, the score (higher is better) should be:
    # score = -(mu - kappa * sigma) = -mu + kappa * sigma
    
    # Adaptive kappa:
    # We can tune kappa based on the scale of sigma and mu.
    # A common heuristic is to use the mean of sigma or a quantile.
    # Let's use a fixed kappa that works well for the Weierstrass function.
    # Given the range of x is [-0.5, 0.5] and the function values are typically O(1),
    # a kappa around 1.0 to 3.0 is reasonable.
    # We can also make it adaptive: kappa = 1.0 + 0.5 * (mean_sigma / max_sigma) ?
    # Let's start with a simple adaptive scheme based on the current uncertainty level.
    
    # Normalize sigma to get a relative exploration weight
    # If all sigmas are small, we can afford more exploitation (lower kappa).
    # If sigmas are large, we need more exploration (higher kappa).
    # However, since sigma is in objective units, we can just use a fixed kappa
    # or one that scales with the magnitude of sigma.
    
    # Let's try a fixed kappa first, as it's simpler and stable.
    # kappa = 2.0 is a standard choice for UCB.
    kappa = 2.0
    
    # Calculate UCB scores
    # Score = -mu + kappa * sigma
    scores = -mu + kappa * sigma
    
    # Check deadline
    if time.time() > deadline:
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
