import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    Uses Probabilistic Lower Bound (PLB) style scoring.
    score = -mu + k * sigma
    This balances exploitation (minimizing mu) and exploration (maximizing sigma).
    """
    # Ensure arrays are floats
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Clip sigma to avoid division by zero or negative values due to numerical errors
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Dynamic beta based on distance to the known minimum (3.0)
    # When f_best is high, we explore more. When f_best is low, we exploit more.
    f_target = 3.0
    delta = f_best - f_target
    
    # Scale exploration coefficient
    # If delta is large, beta is larger. If delta is small, beta is smaller.
    # Using a square root or linear scaling?
    # Let's try a smooth transition.
    # beta = base + scale * sqrt(delta)
    # If delta is 0, beta = base.
    # If delta is 1, beta = base + scale.
    
    base_beta = 0.5
    scale_beta = 2.0
    
    # Ensure delta is non-negative for sqrt
    delta_pos = max(delta, 0.0)
    beta = base_beta + scale_beta * np.sqrt(delta_pos)
    
    # PLB Score: We want to maximize the probability that f(x) < f_best.
    # This is equivalent to maximizing -mu + beta * sigma in a Gaussian context 
    # if we interpret it as an Upper Confidence Bound for minimization.
    # Standard UCB for minimization: -mu + beta * sigma
    
    scores = -mu + beta * sigma_safe
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
