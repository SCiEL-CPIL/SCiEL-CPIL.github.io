"""Cost-aware acquisition using Expected Improvement (EI) for Griewank 2D BO."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI) for
    Griewank 2D Bayesian Optimization.
    
    This function computes the Expected Improvement of each candidate
    relative to the current best observed value, normalized by the cost
    of evaluation. EI naturally balances exploitation (low mu) and
    exploration (high sigma), promoting diversity in the search space
    by favoring regions with high predictive uncertainty.
    
    Parameters:
    mu : array-like, shape (N,)
        Mean of the GP posterior.
    sigma : array-like, shape (N,)
        Standard deviation of the GP posterior.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (N,)
        Cost of evaluating each candidate point.
        
    Returns:
    scores : array, shape (N,)
        Acquisition scores (higher is better).
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is positive and finite
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clip sigma to avoid division by zero in z calculation
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Check for time budget
    elapsed = time.time() - start_time
    if elapsed > 250:
        # Fallback: simple EI approximation or LCB if scipy fails or time is tight
        # Using LCB as a fast fallback
        beta = 1.5
        lcb_score = beta * sigma_safe - mu
        score = lcb_score / cost_safe
        score = np.where(np.isfinite(score), score, 0.0)
        return score

    # Calculate Expected Improvement (EI) for minimization
    # Improvement = f_best - f(x)
    # EI = E[max(f_best - f(x), 0)]
    
    # z = (f_best - mu) / sigma
    # To avoid overflow/underflow in norm.cdf/pdf for extreme z, clip z
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10) # Numerical stability
    
    # EI = sigma * (z * Phi(z) + phi(z))
    # where Phi is CDF and phi is PDF of standard normal
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = sigma_safe * (z * cdf_z + pdf_z)
    
    # Ensure EI is non-negative (numerical errors can cause small negatives)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware normalization: EI per unit cost
    # This encourages finding improvements efficiently
    score = ei / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
