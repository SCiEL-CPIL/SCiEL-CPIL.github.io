"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing Powell 4D.
    Uses a cost-aware Expected Improvement (EI) with adaptive exploration,
    combined with a strong cost discount to prefer cheaper evaluations.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Standardize f_best relative to current predictions
    # EI for minimization: E[f_best - f(x) | f(x) < f_best]
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI: divide by cost to prefer cheaper points
    # Use a power to amplify cost preference
    alpha = 1.0
    cost_factor = np.power(cost_safe, alpha)
    
    # Primary score: cost-aware EI
    score = ei / cost_factor
    
    # Add a small LCB component for exploration in uncertain regions
    # LCB: mu - kappa * sigma (lower is better, so negate for maximization)
    kappa = 0.5
    lcb = mu - kappa * sigma_safe
    lcb_score = -lcb
    
    # Scale LCB to be comparable to EI
    ei_scale = np.std(ei) + 1e-8
    lcb_scale = np.std(lcb_score) + 1e-8
    
    # Blend: 70% cost-aware EI, 30% LCB
    score = 0.7 * (ei / (ei_scale + 1e-8)) + 0.3 * (lcb_score / (lcb_scale + 1e-8))
    
    # Apply cost penalty to the combined score
    score = score / cost_factor
    
    # Ensure all outputs are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
