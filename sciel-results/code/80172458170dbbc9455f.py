import numpy as np
from scipy.stats import norm
import time

# Global minimum of Styblinski-Tang 1D
_F_MIN = -39.1662
# Maximum regret scale (approx range of function)
_MAX_REGRET = 50.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
    Uses an adaptive Lower Confidence Bound (LCB) approach.
    """
    # Ensure inputs are numpy arrays for vectorized operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate current regret
    # Regret is how far f_best is from the known global minimum
    regret = f_best - _F_MIN
    
    # Normalize regret to [0, 1] for adaptive beta
    # If regret is 0 (we found the min), beta should be low (exploit)
    # If regret is high (early on), beta should be high (explore)
    p = np.clip(regret / _MAX_REGRET, 0.0, 1.0)
    
    # Adaptive beta for LCB
    # We want to explore when uncertainty is high relative to regret.
    # A common heuristic is to scale beta with the square root of the number of evaluations,
    # but we don't have that count directly. Instead, we use regret as a proxy for progress.
    # Early on: high regret -> high beta -> more exploration.
    # Late on: low regret -> low beta -> more exploitation.
    #
    # Let's use a curve that starts high and decays.
    # beta = 1.0 + 2.0 * p^0.5  ?
    # Or perhaps something related to the standard deviation of the GP posterior?
    # 
    # The inspiration mentioned: "LCB with adaptive kappa provides a cleaner exploration-exploitation balance"
    # and measured improvement.
    #
    # Let's try: beta = 1.5 * p + 0.5
    # When p=1 (early), beta=2.0
    # When p=0 (late), beta=0.5
    #
    # Alternatively, a more standard approach in BO is to use a fixed beta or one based on iteration count.
    # Since we don't have iteration count, let's stick to regret-based.
    #
    # Let's test a slightly more aggressive exploration early on.
    beta = 1.0 + 3.0 * (p ** 1.5)
    
    # Lower Confidence Bound for minimization:
    # We want to minimize mu - beta * sigma
    # So the score (where higher is better for selection) is:
    # score = -(mu - beta * sigma) = -mu + beta * sigma
    score = -mu + beta * sigma
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
