"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement and
    Lower Confidence Bound (optimistic) terms, both normalized by cost.
    
    The strategy balances:
    - Exploitation: regions where mu is close to (or below) f_best
    - Exploration: regions with high uncertainty (sigma)
    - Cost awareness: prefer cheaper evaluations
    
    We use a hybrid: alpha * EI/cost + (1-alpha) * LCB/cost
    where LCB = f_best - mu + beta * sigma (optimistic gap)
    and beta is scaled to encourage exploration in high-variance regions.
    
    mu, sigma, f_best, cost all have shape (2000,).
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Guard against division by zero and non-finite values
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI is non-negative by definition
    
    # Lower Confidence Bound (optimistic): we want to maximize this
    # f_best - mu + beta * sigma
    # beta is chosen to balance exploration; scale by median sigma for robustness
    sigma_median = np.median(sigma)
    if sigma_median > 0:
        beta = 2.0  # Moderate exploration coefficient
    else:
        beta = 0.0
    
    lcb = (f_best - mu) + beta * sigma_safe
    lcb = np.maximum(lcb, 0.0)  # Non-negative optimistic gap
    
    # Cost-normalized scores
    ei_score = ei / cost_safe
    lcb_score = lcb / cost_safe
    
    # Hybrid: weight EI more when uncertainty is low, LCB more when high
    # Use a simple fixed blend for stability
    alpha = 0.5
    score = alpha * ei_score + (1.0 - alpha) * lcb_score
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
