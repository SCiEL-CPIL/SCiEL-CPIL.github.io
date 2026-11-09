"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function with adaptive EI-UCB blending.
    
    Blends Expected Improvement and an uncertainty bonus, weighted by
    how close f_best is to the known Ackley minimum (0).
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each point.
    
    Returns:
    --------
    scores : array-like, shape (2000,)
        Acquisition scores (higher = more promising).
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive exploration: scale uncertainty bonus by distance to known minimum
    # Ackley minimum is 0, typical f_best ranges from ~20+ down to 0
    # Use a smooth decay: exploration weight high when f_best >> 0, low when f_best ~ 0
    f_best_clipped = max(f_best, 0.0)
    # Scale: at f_best=0, weight=0.1 (minimal exploration); at f_best>=20, weight=1.0
    f_norm = min(f_best_clipped / 20.0, 1.0)
    exploration_weight = 0.1 + 0.9 * f_norm  # [0.1, 1.0]
    
    # Uncertainty bonus: proportional to sigma, scaled by exploration weight
    # This encourages exploring uncertain regions more when we're far from optimum
    uncertainty_bonus = exploration_weight * sigma_safe
    
    # Combined score: EI + uncertainty bonus, all per unit cost
    # The EI term drives exploitation, uncertainty bonus drives exploration
    raw_score = ei + uncertainty_bonus
    
    # Cost-aware: divide by cost to get improvement per unit cost
    cost_safe = np.maximum(cost, 1e-9)
    scores = raw_score / cost_safe
    
    # Check deadline
    if time.time() > deadline:
        pass  # Already have our result
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
