# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Styblinski-Tang 2D.
    
    Uses a hybrid of cost-normalized Expected Improvement (EI) and cost-normalized
    Lower Confidence Bound (LCB) to balance exploitation and exploration while
    accounting for evaluation cost.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        Mean of GP posterior at 2000 candidate points
    sigma : array_like, shape (2000,)
        Standard deviation of GP posterior at 2000 candidate points
    f_best : float
        Best observed objective value so far
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x scaled to [0,1]))
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: handle edge cases
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Lower Confidence Bound (optimistic prediction)
    # Use a moderate kappa for exploration
    kappa = 1.5
    lcb = mu - kappa * sigma_safe
    
    # Cost-normalized versions
    # For EI: higher EI per unit cost is better
    ei_cost = ei / cost_safe
    
    # For LCB: lower LCB is better, so we negate to make higher = better
    # Then normalize by cost
    lcb_cost = -lcb / cost_safe
    
    # Hybrid: combine EI and LCB components
    # Weight them to balance exploitation (EI) and exploration (LCB)
    # As optimization progresses, we want more exploitation, but we don't know
    # the iteration count, so use a fixed blend
    weight_ei = 0.6
    weight_lcb = 0.4
    
    scores = weight_ei * ei_cost + weight_lcb * lcb_cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
