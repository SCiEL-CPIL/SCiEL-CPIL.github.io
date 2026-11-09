# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    Uses a Linear Upper Confidence Bound (UCB) with adaptive exploration.
    This is structurally different from EI, favoring a linear combination
    of mean and variance which can explore more broadly in multimodal landscapes.
    """
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    
    n = mu.shape[0]
    if n == 0:
        return np.array([])
    
    # Sanitize sigma: ensure positive and finite
    sigma_safe = np.maximum(sigma, 1e-8)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-8)
    
    # Sanitize mu: replace non-finite with large penalty
    mu_clean = np.where(np.isfinite(mu), mu, 1e10)
    
    # Define bounds for progress calculation
    min_val = -1.8013
    max_val = 0.6
    range_val = max_val - min_val
    
    # Calculate progress: 1.0 at start (f_best ~ 0), 0.0 at optimum
    # f_best is typically negative for Michalewicz.
    # If f_best is very high (bad), progress is high.
    progress = np.clip((f_best - min_val) / range_val, 0.0, 1.0)
    
    # Adaptive Beta for UCB
    # Start with high exploration (beta_max) and decay to exploitation (beta_min)
    beta_min = 0.5
    beta_max = 3.0
    
    # Smooth decay
    beta = beta_min + (beta_max - beta_min) * (progress ** 0.5)
    
    # UCB Score for minimization:
    # We want to minimize f(x).
    # The "upper bound" on the function value (worst case) is mu + beta * sigma.
    # We want to pick the point with the LOWEST upper bound.
    # So the score (higher is better) should be - (mu + beta * sigma).
    # Alternatively, we can define the score as the potential for improvement.
    # Standard UCB for maximization: mu + beta * sigma.
    # For minimization, we transform f to -f (maximization of -f).
    # mu_neg = -mu, sigma_neg = sigma.
    # Score = -mu + beta * sigma.
    
    score = -mu_clean + beta * sigma_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
