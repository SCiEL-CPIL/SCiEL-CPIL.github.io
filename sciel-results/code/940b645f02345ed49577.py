# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a modified Expected Improvement
    with cost weighting, combined with a lower confidence bound penalty.
    
    The score balances expected improvement per unit cost with an
    uncertainty-driven exploration term scaled by cost.
    """
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-normalized EI (higher = better per unit cost)
    ei_cost = ei / (cost + 1e-9)
    
    # Lower Confidence Bound component (exploration)
    # Use a fixed beta that balances exploration vs exploitation
    # The LCB is mu - beta * sigma; we want to minimize this, so
    # we use -LCB = -mu + beta * sigma as a score (higher = more uncertain)
    beta = 2.0
    lcb_score = -mu + beta * sigma_safe
    
    # Normalize lcb_score to be comparable to ei_cost scale
    # Use the mean of lcb_score as a scale reference
    lcb_scale = np.mean(np.abs(lcb_score)) + 1e-9
    lcb_normalized = lcb_score / lcb_scale
    
    # Combine: primary signal is cost-normalized EI, secondary is LCB exploration
    # Weight: 70% EI cost, 30% LCB exploration
    score = 0.7 * ei_cost + 0.3 * lcb_normalized
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
