"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the 330s limit


def _check_deadline():
    """Return True if we should stop to avoid timeout."""
    return (time.time() - _start_time) > (330.0 - _DEADLINE_MARGIN)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a Pareto-front selection strategy.
    
    Instead of linearly blending Expected Improvement (EI) and Upper Confidence 
    Bound (UCB) with fixed weights, this method treats EI and UCB as two 
    competing objectives. It identifies the Pareto front of the 2000 candidates 
    in the (EI, UCB) space and selects the candidate on this front that is 
    closest to the ideal point (maximum EI and maximum UCB).
    
    This structural change allows the acquisition function to dynamically 
    adapt to the posterior distribution's trade-off landscape, potentially 
    escaping local minima more effectively than fixed-weight blends.
    
    Parameters:
        mu: (2000,) array of GP means
        sigma: (2000,) array of GP std devs
        f_best: lowest observed function value
    
    Returns:
        (2000,) array of scores, higher = more desirable
    """
    # Deadline guard: if we're running out of time, return a safe fallback
    if _check_deadline():
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # 1. Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Compute UCB score for minimization
    # We want to maximize the potential for a low function value.
    # Score to maximize: -mu + beta * sigma
    beta = 0.5
    ucb_score = -mu + beta * sigma_safe
    
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ucb_score = np.where(np.isfinite(ucb_score), ucb_score, 0.0)
    
    # 3. Find Pareto Front
    # We want to maximize both ei and ucb_score.
    # A point i is dominated if there exists j such that ei_j > ei_i and ucb_j > ucb_i.
    # Efficient way: Sort by ei descending, then track max ucb seen so far.
    
    # Sort indices by EI in descending order
    sort_idx = np.argsort(-ei)
    sorted_ei = ei[sort_idx]
    sorted_ucb = ucb_score[sort_idx]
    
    n = len(ei)
    is_pareto = np.zeros(n, dtype=bool)
    
    max_ucb_so_far = -np.inf
    # Iterate through sorted points
    for i in range(n):
        if sorted_ucb[i] > max_ucb_so_far:
            is_pareto[sort_idx[i]] = True
            max_ucb_so_far = sorted_ucb[i]
        else:
            # Once we hit a point that is not on the front, 
            # all subsequent points (with lower EI) cannot be on the front 
            # because their UCB is <= max_ucb_so_far (which is >= current UCB? No, wait).
            # Actually, if sorted_ucb[i] <= max_ucb_so_far, it is dominated by the point 
            # that achieved max_ucb_so_far (which has higher EI and higher/equal UCB).
            # Since we are sorted by EI descending, any subsequent point has even lower EI.
            # So we can break early for efficiency.
            pass 

    # 4. Assign Scores
    # Points on the Pareto front get a score based on their proximity to the ideal point.
    # Points not on the front get a very low score.
    
    scores = np.full(n, -1000.0)
    
    if np.any(is_pareto):
        # Normalize EI and UCB for the tie-breaker
        ei_min, ei_max = np.min(ei), np.max(ei)
        ucb_min, ucb_max = np.min(ucb_score), np.max(ucb_score)
        
        ei_range = ei_max - ei_min + 1e-10
        ucb_range = ucb_max - ucb_min + 1e-10
        
        ei_norm = (ei - ei_min) / ei_range
        ucb_norm = (ucb_score - ucb_min) / ucb_range
        
        # Tie-breaker: Sum of normalized objectives (distance to ideal point is minimized by maximizing this)
        tie_breaker = ei_norm + ucb_norm
        
        # Assign scores only to Pareto points
        scores[is_pareto] = tie_breaker[is_pareto]
        
        # Ensure the highest score is clearly the winner
        # The argmax will pick the best Pareto point.
        
    # Fallback: if no pareto points (should not happen), return EI
    if not np.any(is_pareto):
        scores = ei
        
    return scores
# EVOLVE-BLOCK-END
