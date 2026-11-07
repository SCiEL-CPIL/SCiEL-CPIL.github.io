"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def _quantile_of_improvement(mu, sigma, f_best):
    """
    Quantile of Improvement (QI) acquisition function.
    
    QI maximizes the conditional expected value of the objective given improvement.
    For minimization, we want low values. The conditional expectation of f(x) given f(x) < f_best
    is: E[f(x) | f(x) < f_best] = mu - sigma * phi(z) / Phi(z) where z = (f_best - mu)/sigma.
    
    We want to minimize this, so our acquisition score (higher is better) is:
    score = -E[f(x) | f(x) < f_best]
    
    But we also need cost-awareness. The cost is 1 + mean(x_scaled), which is between 1 and 2.
    We divide by cost to prefer cheaper points.
    
    To avoid numerical issues when Phi(z) is very small (z very negative), we clamp.
    """
    z = (f_best - mu) / (sigma + 1e-12)
    
    # Phi(z) = CDF of standard normal at z
    cdf_z = norm.cdf(z)
    # phi(z) = PDF of standard normal at z
    pdf_z = norm.pdf(z)
    
    # Avoid division by zero
    cdf_z_safe = np.maximum(cdf_z, 1e-12)
    
    # Conditional expectation: E[f | f < f_best] = mu - sigma * phi(z) / Phi(z)
    # This is the mean of the truncated normal below f_best
    conditional_mean = mu - sigma * pdf_z / cdf_z_safe
    
    # We want to minimize conditional_mean, so maximize -conditional_mean
    # Also, we only care about improvement, so if P(f < f_best) is very small, the signal is weak
    # QI naturally handles this: when cdf_z is small, the conditional mean is driven by the
    # far tail, but the "confidence" in improvement is low.
    
    # For a pure QI, we just return -conditional_mean
    # But to be cost-aware: score = -conditional_mean / cost
    
    # Actually, let's think about this more carefully.
    # The standard QI is: QI(x) = E[f(x) | f(x) < f_best]
    # For minimization, lower is better. So we want to MINIMIZE QI(x).
    # The acquisition function should be MAXIMIZED. So we return -QI(x) / cost.
    
    return -conditional_mean

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Quantile of Improvement (QI) acquisition function.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled))
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Start time for deadline guard
    t_start = time.time()
    deadline = t_start + 290.0  # Safety margin: 290s out of 300s
    
    # Compute QI scores
    qi = _quantile_of_improvement(mu, sigma, f_best)
    
    # Cost-aware normalization
    # cost is between 1 and 2
    # Divide by cost to prefer cheaper evaluations
    scores = qi / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Deadline check
    t_elapsed = time.time() - t_start
    if t_elapsed > (300.0 - 10.0):  # If we're close to the limit, return immediately
        return scores
    
    return scores
# EVOLVE-BLOCK-END
