"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: kill program after 330s wall-clock, but we only have acquisition()
# The acquisition function itself needs to be fast.
# The evaluator runs the whole loop, so we just need acquisition to be correct and fast.

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: (2000,) array of GP mean predictions
    - sigma: (2000,) array of GP standard deviations
    - f_best: scalar, lowest observed value
    - cost: (2000,) array of evaluation costs (1 + mean(x) for each candidate)
    
    Returns:
    - (2000,) array of acquisition scores (higher = evaluate next)
    """
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Clip negative EI to 0 (EI should be non-negative when mu < f_best)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost to get improvement per unit cost
    # cost is already 1 + mean(x_scaled), so it's in [1, 2]
    scores = ei / cost
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
