"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Rastrigin 2D.
    
    Combines Expected Improvement (EI) with a UCB exploration term, using
    adaptive weights based on f_best. EI is the primary signal (it scored best
    in isolation), with a small UCB component to help escape local minima in
    the multimodal Rastrigin landscape. All terms are normalized by evaluation
    cost to handle the cost-aware requirement.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions at 2000 candidate points.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions at 2000 candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next).
    """
    t_start = time.time()
    DEADLINE = 290.0  # safety margin before 330s kill

    # Guard against numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score for EI
    improvement = f_best - mu  # positive when mu < f_best
    z = improvement / sigma_safe
    
    # Expected Improvement (positive when mu < f_best)
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB score for minimization: -mu + beta * sigma
    # Higher means more promising (low mu, high uncertainty)
    if f_best > 10.0:
        beta = 2.0
    elif f_best > 5.0:
        beta = 1.5
    elif f_best > 2.0:
        beta = 1.0
    elif f_best > 0.5:
        beta = 0.7
    else:
        beta = 0.5
    
    ucb_score = -mu + beta * sigma_safe
    
    # Normalize EI and UCB to comparable scales
    ei_max = np.max(ei) if np.max(ei) > 1e-10 else 1.0
    ucb_max = np.max(ucb_score) if np.max(ucb_score) > 1e-10 else 1.0
    
    ei_norm = ei / ei_max
    ucb_norm = ucb_score / ucb_max
    
    # Adaptive blend: EI-dominant (it scored best standalone), with UCB
    # providing exploration help in early/mid phases
    if f_best > 10.0:
        ei_weight = 0.6
        ucb_weight = 0.4
    elif f_best > 5.0:
        ei_weight = 0.7
        ucb_weight = 0.3
    elif f_best > 2.0:
        ei_weight = 0.8
        ucb_weight = 0.2
    elif f_best > 0.5:
        ei_weight = 0.9
        ucb_weight = 0.1
    else:
        ei_weight = 0.95
        ucb_weight = 0.05
    
    combined = ei_weight * ei_norm + ucb_weight * ucb_norm
    
    # Cost-aware: divide by cost
    scores = combined / cost_safe
    
    # Small exploration bonus for very early iterations
    if f_best > 8.0:
        explore_bonus_weight = min(0.05, (f_best - 8.0) / 100.0)
        scores = scores + explore_bonus_weight * sigma_safe / cost_safe
    
    # Deadline check
    if time.time() - t_start > DEADLINE - 5.0:
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
