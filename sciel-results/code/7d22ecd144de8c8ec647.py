"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track global start time for deadline guard
_T0 = time.time()
_DEADLINE_MARGIN = 15.0  # seconds before hard limit


def _elapsed():
    return time.time() - _T0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    and UCB (Upper Confidence Bound) with a cost penalty.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviations
        f_best: float - best observed value so far
        cost: array of shape (2000,) - evaluation costs
    
    Returns:
        scores: array of shape (2000,) - higher means evaluate next
    """
    # Deadline guard: if we're running out of time, return a safe fallback
    if _elapsed() > 300.0 - _DEADLINE_MARGIN:
        # Fallback: simple EI without extra computation
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return ei / np.maximum(cost, 1e-6)
    
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB component: exploration bonus
    # Use a moderate beta to balance exploration
    beta = 0.5
    ucb = mu + beta * sigma_safe
    
    # Combine: we want to minimize, so for UCB we want low mu + high uncertainty
    # Actually for minimization, UCB should be: mu - beta * sigma (lower bound)
    # But in acquisition functions for minimization, we typically use:
    # -EI or -UCB where UCB = mu - beta*sigma
    # Let's use a weighted combination
    
    # For minimization:
    # EI is already positive when mu < f_best (improvement expected)
    # UCB for minimization: we prefer points with low mu and high sigma
    # ucb_min = mu - beta * sigma (lower is better for selection)
    # So score_ucb = -ucb_min = -mu + beta * sigma
    
    # Normalize components to comparable scales
    # EI is in objective units, ucb score is also in objective units
    
    # Weighted combination: primarily EI with some exploration
    w_ei = 1.0
    w_ucb = 0.3
    
    score = w_ei * ei + w_ucb * (-mu + beta * sigma_safe)
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    cost_safe = np.maximum(cost, 1e-6)
    scores = score / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
