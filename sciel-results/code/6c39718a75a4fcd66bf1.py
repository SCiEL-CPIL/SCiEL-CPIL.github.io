import numpy as np
from scipy.stats import norm
import time

# Global time tracking for deadline guard
_start_time = time.time()
_DEADLINE_OFFSET = 320.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    mu: Array of shape (2000,) - GP mean
    sigma: Array of shape (2000,) - GP standard deviation
    f_best: Float - Best observed value
    cost: Array of shape (2000,) - Cost of evaluation
    
    Returns:
    Array of shape (2000,) - Acquisition scores (higher is better)
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE_OFFSET:
        # Fallback: Exploit the best mean, adjusted by cost
        # This is a safe, finite return to avoid crashing or invalid scores
        mu_safe = np.asarray(mu)
        cost_safe = np.asarray(cost)
        cost_safe = np.maximum(cost_safe, 1e-9)
        return -mu_safe / cost_safe

    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Ensure sigma and cost are positive to avoid division errors
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # 1. Expected Improvement (EI)
    # EI = E[max(0, f_best - f(x))]
    # For Gaussian f(x) ~ N(mu, sigma^2):
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # 2. Upper Confidence Bound (UCB)
    # For minimization, we maximize: -mu + beta * sigma
    # This encourages low mean (exploitation) and high variance (exploration)
    beta = 1.5
    ucb = -mu + beta * sigma_safe
    
    # 3. Combine and Normalize by Cost
    # Blending EI and UCB balances exploitation and exploration
    # Dividing by cost favors cheaper evaluations
    score = (ei + ucb) / cost_safe
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
