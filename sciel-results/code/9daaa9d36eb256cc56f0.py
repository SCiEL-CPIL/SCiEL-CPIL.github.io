"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Cost-Adjusted Expected Improvement (CAEI).
    
    This function computes the Expected Improvement (EI) for each candidate 
    and divides it by the evaluation cost. This prioritizes candidates that 
    offer the highest expected reduction in the objective function per unit 
    of cost.
    
    Parameters:
        mu: array of shape (2000,), GP mean predictions
        sigma: array of shape (2000,), GP standard deviation predictions
        f_best: float, best observed function value
        cost: array of shape (2000,), cost of evaluating each candidate
    
    Returns:
        array of shape (2000,), acquisition scores (higher = evaluate next)
    """
    # Ensure inputs are numpy arrays of float64
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against division by zero or negative costs
    # Costs are 1 + mean(scaled_x), so min cost is 1.0. 
    # But we guard just in case.
    cost_safe = np.maximum(cost, 1e-9)
    
    # Guard against zero/negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is CDF and phi is PDF of standard normal distribution
    
    # To avoid numerical issues with very large/small z:
    # If z is very large positive, Phi(z) ~ 1, phi(z) ~ 0. EI ~ f_best - mu.
    # If z is very large negative, Phi(z) ~ 0, phi(z) ~ 0. EI ~ 0.
    
    # Compute Phi(z) and phi(z)
    # norm.cdf(z) and norm.pdf(z)
    
    # Optimization: Use vectorized operations
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Calculate EI
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # For points where sigma was effectively 0 (or very small), 
    # the EI formula above might suffer from numerical noise or 
    # incorrect limits if sigma_safe was clamped.
    # If sigma < epsilon, the posterior is a delta function at mu.
    # EI should be max(0, f_best - mu).
    # Let's identify low-sigma regions
    low_sigma_mask = sigma < 1e-6
    
    # Calculate deterministic improvement for low sigma regions
    # If mu < f_best, improvement is f_best - mu. Else 0.
    deterministic_ei = np.maximum(0.0, f_best - mu)
    
    # Combine
    ei = np.where(low_sigma_mask, deterministic_ei, ei)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-Adjusted Score
    score = ei / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
