import numpy as np
import time
from scipy.stats import norm

# Global timer for safety
_start_time = time.time()
_DEADLINE = 320.0  # 10s margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Uses Upper Confidence Bound (UCB) for minimization.
    Score = -mu + beta * sigma
    
    The Weierstrass function is rugged and oscillatory. UCB with a moderate
    beta encourages exploration of unexplored regions (high sigma) while
    still favoring regions with low predicted mean (-mu). This is a direct
    and robust approach for such landscapes.
    """
    # Check time budget
    if time.time() - _start_time > _DEADLINE:
        return np.zeros_like(mu)

    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-10)
    
    # UCB parameter: beta controls the exploration/exploitation trade-off
    # Higher beta -> more exploration
    beta = 2.5
    
    # UCB score for minimization:
    # We want to find points with low mu (good) and high sigma (uncertain)
    # score = -mu + beta * sigma
    score = -mu + beta * sigma
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
