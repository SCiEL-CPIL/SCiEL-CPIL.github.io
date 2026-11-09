"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Rastrigin 2D.
    
    Uses adaptive blending of Expected Improvement (EI) and Upper Confidence
    Bound (UCB) with f_best-dependent weights, plus a small uncertainty bonus.
    
    For the multimodal Rastrigin landscape, the exploration weight is tuned to help
    escape local minima while exploiting cheap promising points.
    
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
    DEADLINE = 300.0  # seconds, safety margin before 330s kill
    
    # Guard against numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score for EI
    improvement = f_best - mu  # positive when mu < f_best
    z = improvement / sigma_safe
    
    # Expected Improvement (positive when mu < f_best)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB: mu + beta * sigma (for minimization, we want low mu + high uncertainty)
    # For minimization, UCB-like score is -mu + beta*sigma
    # We use a beta that adapts based on how close f_best is to the minimum
    # Rastrigin min is 0, typical values up to ~20
    
    # Adaptive beta: larger when f_best is far from minimum (early exploration)
    # beta ranges from ~0.5 (late, exploitation) to ~2.0 (early, exploration)
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
    
    # UCB score for minimization: higher is better when mu is low and sigma is high
    ucb_score = -mu + beta * sigma_safe
    
    # Normalize EI and UCB to comparable scales
    ei_max = np.max(ei) if np.max(ei) > 1e-10 else 1.0
    ucb_max = np.max(ucb_score) if np.max(ucb_score) > 1e-10 else 1.0
    
    ei_norm = ei / ei_max
    ucb_norm = ucb_score / ucb_max
    
    # Adaptive blend: more EI when f_best is close to 0, more UCB when far
    if f_best > 5.0:
        # Early: more exploration via UCB
        ei_weight = 0.3
        ucb_weight = 0.7
    elif f_best > 2.0:
        # Mid: balanced
        ei_weight = 0.5
        ucb_weight = 0.5
    elif f_best > 0.5:
        # Late: more exploitation via EI
        ei_weight = 0.7
        ucb_weight = 0.3
    else:
        # Very late: mostly exploitation
        ei_weight = 0.8
        ucb_weight = 0.2
    
    # Combined score
    combined = ei_weight * ei_norm + ucb_weight * ucb_norm
    
    # Cost-aware: divide by cost
    scores = combined / cost_safe
    
    # Small uncertainty bonus for very early iterations
    if f_best > 8.0:
        explore_bonus_weight = min(0.05, (f_best - 8.0) / 100.0)
        scores = scores + explore_bonus_weight * sigma_safe / cost_safe
    
    # Deadline check
    if time.time() - t_start > DEADLINE - 5.0:
        # Return best result found so far with safety margin
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
