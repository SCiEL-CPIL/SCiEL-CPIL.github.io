"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition combining Expected Improvement (EI) and Lower Confidence Bound (LCB).
    
    The score balances:
    1. EI: rewards candidates that are likely to improve upon f_best
    2. LCB: rewards candidates with low predicted values and high uncertainty
    3. Cost normalization: penalizes expensive evaluations
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions for 2000 candidates
    sigma : array-like, shape (2000,)
        GP standard deviation predictions for 2000 candidates
    f_best : float
        Best observed value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive and handle numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Lower Confidence Bound (LCB) - negative because we want to minimize
    # beta controls exploration: larger beta = more exploration
    beta = 2.0
    lcb = mu - beta * sigma_safe
    
    # Combine EI and LCB
    # EI is in objective units (positive when improvement expected)
    # LCB is in objective units (lower is better)
    # We want higher scores to be better, so negate LCB
    # Scale EI and LCB to comparable ranges
    # EI can be 0 to (f_best - min_mu)
    # -LCB can be -max(mu - beta*sigma) to -min(mu - beta*sigma)
    
    # Normalize by typical scales
    # Use a weighted combination
    # EI is already scaled by f_best
    # For LCB, we want to reward low values, so use -lcb
    # But we need to make them comparable
    
    # Strategy: use EI as the primary signal, add LCB as a secondary signal
    # Scale LCB contribution relative to EI scale
    ei_scale = max(np.max(ei), 1e-9)
    lcb_scale = max(np.max(-lcb), 1e-9)
    
    # Combine: weight EI more heavily, add LCB as exploration bonus
    weight_ei = 0.7
    weight_lcb = 0.3
    
    # Normalize each component
    ei_norm = ei / ei_scale
    lcb_norm = (-lcb) / lcb_scale
    
    # Combine normalized scores
    combined = weight_ei * ei_norm + weight_lcb * lcb_norm
    
    # Normalize by cost
    cost_safe = np.maximum(cost, 1e-9)
    scores = combined / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
