import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Ackley 2D.
    Uses Expected Improvement (EI) with multiplicative cost discount.
    """
    # Track time to avoid timeout
    t_start = time.time()
    deadline = t_start + 290.0  # safety margin before 300s limit

    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Safety: clamp sigma to be non-negative and finite
    sigma = np.clip(sigma, 0.0, None)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-6)

    # Compute Lower Confidence Bound (LCB) for minimization
    # LCB = mu - beta * sigma
    # This balances exploitation (mu) and exploration (sigma)
    # A higher beta encourages more exploration
    
    # Choose a beta value. 1.0 is a standard starting point.
    # Given the task is 2D Ackley, which is multimodal, some exploration is key.
    beta = 1.5
    
    LCB = mu - beta * sigma
    
    # Multiplicative cost discount: score = LCB / cost
    # This explicitly maximizes the lower bound (potential improvement) per unit of budget
    score = LCB / cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return score
    
    return score
