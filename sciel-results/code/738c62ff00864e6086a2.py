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
        # Fallback: simple cost-normalized exploit
        mu_safe = np.asarray(mu, dtype=np.float64)
        cost_safe = np.asarray(cost, dtype=np.float64)
        cost_safe = np.maximum(cost_safe, 1e-9)
        return -mu_safe / cost_safe

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values for safety
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma and cost are positive to avoid division errors
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    # Clip z for numerical stability
    z = np.clip(z, -50, 50)
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Upper Confidence Bound (UCB) for minimization
    # Adaptive beta: start high for exploration, decay as we find better solutions
    # f_best starts high (rastrigin min is 0, max ~40), so use normalized progress
    # When f_best is close to 0, we should exploit more
    f_best_clamped = np.clip(f_best, 0.0, 50.0)
    progress = 1.0 - (f_best_clamped / 50.0)  # 0 when f_best=50, 1 when f_best=0
    
    # Beta ranges from 2.0 (early) to 1.0 (late)
    beta = 2.0 * (1.0 - 0.5 * progress)
    ucb = -mu + beta * sigma_safe
    
    # Weighted combination:
    # - EI: strong exploitation signal
    # - UCB: exploration-exploitation balance  
    # - PI: early exploration boost
    # 
    # Structure: blend EI and UCB with cost normalization
    # Add a small PI term to encourage exploration when improvement is likely
    w_ei = 1.0
    w_ucb = 0.5
    w_pi = 0.1 * (1.0 - progress)  # PI weight decays as we progress
    
    score = (w_ei * ei + w_ucb * ucb + w_pi * cdf_z * sigma_safe) / cost_safe
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
