import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Quantile of Improvement (QI) acquisition function with cost awareness.
    
    QI focuses on high-probability, high-reward regions by maximizing the
    conditional expected improvement, which better balances exploration and
    exploitation in multimodal landscapes like Rastrigin.
    
    Args:
        mu: (2000,) array of GP means.
        sigma: (2000,) array of GP standard deviations.
        f_best: float, the lowest observed value.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of scores (higher is better).
    """
    # Deadline guard: track elapsed time
    t_start = time.time()
    deadline = 300.0  # 300s limit per evaluation
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Quantile of Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Quantile of Improvement:
    # QI = E[f | f < f_best] = mu - sigma * phi(z) / (1 - Phi(z))
    # where Phi is CDF, phi is PDF of standard normal
    # 
    # For numerical stability, handle cases where z is very negative or positive
    # When z is very negative (mu >> f_best), 1 - Phi(z) ~ 1, so QI ~ mu - sigma * phi(z)
    # When z is very positive (mu << f_best), 1 - Phi(z) is tiny, need care
    
    phi_z = norm.pdf(z)
    one_minus_cdf = 1.0 - norm.cdf(z)
    
    # Clip one_minus_cdf to avoid division by zero
    one_minus_cdf_safe = np.maximum(one_minus_cdf, 1e-12)
    
    # QI = mu - sigma * phi(z) / (1 - Phi(z))
    # This is the expected value of f conditional on f < f_best
    # We want to minimize this, so we negate for maximization
    qi = mu - sigma_safe * phi_z / one_minus_cdf_safe
    
    # For points where f_best - mu is very negative (mu >> f_best),
    # the conditional expectation is dominated by mu, which is bad
    # For points where f_best - mu is very positive (mu << f_best),
    # QI approaches mu, which is good
    # 
    # The QI score: we want to maximize (f_best - QI) to get improvement
    # Actually, QI itself is the expected value given improvement
    # Lower QI is better (we want low function values)
    # So score = -QI, or equivalently, we can use f_best - QI as improvement
    
    # Use f_best - QI as the improvement metric
    # This is always >= 0 when there's a chance of improvement
    improvement = f_best - qi
    
    # Ensure non-negative (numerical stability)
    improvement = np.maximum(improvement, 0.0)
    
    # Cost-aware score: Maximize improvement per unit cost
    # cost is >= 1.0 based on problem description
    score = improvement / cost
    
    # Ensure all values are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check deadline
    t_elapsed = time.time() - t_start
    if t_elapsed > deadline - 5.0:
        # Safety margin: return current result
        return score
    
    return score
