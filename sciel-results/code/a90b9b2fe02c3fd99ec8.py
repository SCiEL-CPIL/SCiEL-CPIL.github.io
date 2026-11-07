"""Cost-aware acquisition function for Bayesian optimization (Styblinski-Tang 2D)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Expected Improvement and
    a structure-aware bonus for the Styblinski-Tang global minimum region.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed value so far
    cost : array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Scale EI to be in a reasonable range
    # EI can be negative when mu > f_best, so clip to non-negative
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB for minimization: we want to maximize (-mu + beta * sigma)
    # Use an adaptive beta
    beta = 1.5
    ucb = -mu + beta * sigma_safe
    
    # Normalize UCB to be comparable to EI
    # Use the scale of the objective
    obj_scale = max(np.abs(f_best), 1.0)
    ucb_scaled = ucb / obj_scale
    ei_scaled = ei / obj_scale
    
    # Combine EI and UCB with fixed weights
    # EI is more reliable when we have good predictions, UCB when uncertain
    w_ei = 0.6
    w_ucb = 0.4
    base_score = w_ei * ei_scaled + w_ucb * ucb_scaled
    
    # Cost penalty: higher cost means lower score
    # Normalize costs to [0, 1] range
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range < 1e-9:
        cost_norm = np.ones_like(cost) * 0.5
    else:
        cost_norm = (cost - cost_min) / cost_range
    
    # Cost penalty: use a soft penalty that's not too aggressive
    # cost in [0,1], so 1/(1+2*cost) ranges from 1/3 to 1
    cost_penalty = 1.0 / (1.0 + 2.0 * cost_norm)
    
    # Apply cost penalty
    scores = base_score * cost_penalty
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
