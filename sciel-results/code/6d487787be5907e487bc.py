"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) acquisition function.
    
    For minimization, we use the Lower Confidence Bound (LCB) of the function value:
    LCB(x) = mu(x) - beta * sigma(x)
    
    We want to maximize the expected improvement over the current best f_best.
    Approximate improvement: f_best - LCB(x) = f_best - mu(x) + beta * sigma(x)
    
    We normalize by the cost to account for the variable evaluation cost.
    Higher score means more likely to be selected.
    """
    # Clip sigma to avoid numerical issues, though GP sigma should be positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Beta controls the trade-off between exploitation (mu) and exploration (sigma)
    # In high dimensions, a higher beta encourages exploration of unvisited regions.
    beta = 1.5
    
    # Calculate the potential improvement using the Lower Confidence Bound
    # If mu is low and sigma is high, the LCB is very low, implying a high chance of a new minimum.
    lcb = mu - beta * sigma_safe
    
    # Improvement over the current best
    improvement = f_best - lcb
    
    # Normalize by cost
    # cost is given as (2000,), f_best is scalar
    scores = improvement / cost
    
    # Ensure all scores are finite
    # If any are NaN or Inf, replace them with a very low value (unlikely to be selected)
    # or a median value to be safe.
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # To prevent extreme values from dominating due to very low cost or high uncertainty,
    # we might clip the scores. However, the evaluator just picks the argmax.
    # Let's add a small penalty for very high scores to ensure stability? No, just return them.
    
    return scores
# EVOLVE-BLOCK-END
