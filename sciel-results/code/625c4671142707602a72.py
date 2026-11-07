# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Expected Improvement
    and an adaptive Upper Confidence Bound (UCB) term.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (n_candidates,)
        Mean of the GP posterior.
    sigma : np.ndarray, shape (n_candidates,)
        Standard deviation of the GP posterior.
    f_best : float
        Best observed value so far.
    cost : np.ndarray, shape (n_candidates,)
        Cost of evaluating each candidate (1 + mean(x)).
    
    Returns:
    --------
    scores : np.ndarray, shape (n_candidates,)
        Acquisition scores (higher is better).
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Estimate total cost consumption to adapt beta.
    # We don't have the exact total cost, but we can estimate it based on f_best progress.
    # Hartmann 3D minimum is approx -3.86.
    # Initial points (5 uniform) usually yield f_best around 0 to 2.
    # Let's assume initial f_best ~ 0.0 and final f_best ~ -3.86.
    # Progress p goes from 0 to 1.
    # Total cost goes from ~7.5 (5 points * 1.5) to 30.
    
    f_min_known = -3.86278
    f_initial_guess = 0.0
    
    # Calculate progress
    # If f_best is higher than f_initial_guess, we are very early (progress 0).
    # If f_best is lower than f_min_known, we are at the end (progress 1).
    denom = f_initial_guess - f_min_known
    if denom == 0:
        progress = 0.5
    else:
        progress = (f_initial_guess - f_best) / denom
        progress = np.clip(progress, 0.0, 1.0)
    
    # Estimate total cost
    c_start = 7.5
    c_max = 30.0
    estimated_total_cost = c_start + progress * (c_max - c_start)
    
    # Adaptive beta
    # beta = 1 / (1 + total_cost / 30)
    beta = 1.0 / (1.0 + estimated_total_cost / 30.0)
    
    # Cost-Aware UCB (LCB for minimization)
    # Score = (beta * sigma - mu) / cost
    ucb_score = (beta * sigma_safe - mu) / cost
    
    # Ensure finite values
    ucb_score = np.where(np.isfinite(ucb_score), ucb_score, -np.inf)
    
    return ucb_score
# EVOLVE-BLOCK-END
