"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: keep a safety margin before the 330s wall-clock limit.
    # The function is called repeatedly inside a loop; we track cumulative time
    # to avoid exceeding the budget. If we get close to the deadline, we return
    # a simple deterministic score to finish quickly.
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Clip f_best to [0, 1] for stable interpolation
    f_best_clipped = np.clip(f_best, 0.0, 1.0)
    
    # Structural change: Use pure LCB (Lower Confidence Bound) for minimization
    # LCB = mu - beta * sigma
    # We want to maximize the score, so we negate: score = -mu + beta * sigma
    # This is equivalent to UCB for maximization, or LCB for minimization.
    
    # Adaptive beta:
    # Start with high exploration (high beta) and decay as f_best improves.
    # f_best is in [0, 1] for Griewank.
    # When f_best is high (e.g., 0.5), we are far from optimum, explore more.
    # When f_best is low (e.g., 0.01), we are close, exploit more.
    
    # Use a linear decay from beta_max to beta_min as f_best goes from 1 to 0.
    # Or perhaps a more aggressive exploration early on.
    # Let's try beta = beta_max * (1 - f_best) + beta_min
    # If f_best=1, beta=beta_min. If f_best=0, beta=beta_max.
    # Wait, f_best is the *best* value found.
    # Initially f_best is high (bad). We want high beta.
    # Later f_best is low (good). We want low beta.
    
    # So beta should be high when f_best is high.
    # beta = beta_min + (beta_max - beta_min) * f_best_clipped
    
    beta_min = 1.0
    beta_max = 4.0
    beta = beta_min + (beta_max - beta_min) * f_best_clipped
    
    # LCB score for minimization (negated for maximization):
    # We want to find points where mu is low and sigma is high.
    # score = -mu + beta * sigma
    
    # Cost-awareness:
    # Divide by cost.
    # cost = 1 + mean(x_scaled). Range [1, 2].
    # Dividing by cost penalizes expensive points.
    
    scores = (-mu + beta * sigma) / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
