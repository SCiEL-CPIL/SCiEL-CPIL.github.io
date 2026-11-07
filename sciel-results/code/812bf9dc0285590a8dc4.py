"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills after 330s wall-clock.
# Each acquisition call should be fast, but we add a safety margin.
_DEADLINE = 320.0  # seconds from process start (conservative margin)
_START_TIME = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best (lowest) observed function value
    cost : array of shape (2000,)
        Cost of evaluating each candidate = 1 + mean(x_scaled_to_[0,1])
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Deadline check: if we're getting close to the kill time, return a safe default
    if time.time() - _START_TIME > _DEADLINE:
        # Return uniform scores to avoid invalid results
        return np.ones_like(mu)
    
    # Clamp sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-10)
    
    # Adaptive beta: shift from exploration to exploitation as f_best decreases
    # Rosenbrock 2D minimum is 0. Early on f_best is large, late it approaches 0.
    # Use a smooth transition based on f_best.
    # When f_best is large (early), beta ~ 2.0 (more exploration)
    # When f_best is small (late), beta ~ 0.8 (more exploitation)
    # Transition: beta = 0.8 + 1.2 * sigmoid(-f_best / scale)
    # But we need to be careful: f_best starts around ~5-10 and goes toward 0
    
    # A simpler adaptive scheme: use the magnitude of f_best to modulate exploration
    # beta = 0.8 + 1.2 * min(1.0, max(0.0, f_best / 10.0))
    # When f_best > 10, beta = 2.0; when f_best < 0, beta = 0.8
    
    f_best_val = float(f_best)
    if f_best_val <= 0:
        beta = 0.8
    else:
        # Smooth transition: beta goes from 2.0 at f_best=10 to 0.8 at f_best=0
        beta = 0.8 + 1.2 * min(1.0, f_best_val / 10.0)
    
    # Standardized z-score
    z = (f_best_val - mu) / sigma
    
    # Expected Improvement (EI) with adaptive beta
    # EI_beta = (f_best - mu) * Phi(z) + sigma * phi(z)
    # But we can modify it to be more exploratory by adding a term
    
    # Compute standard EI
    ei = (f_best_val - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add an exploration bonus proportional to sigma^beta
    # This encourages exploration in high-uncertainty regions
    # The beta parameter controls how much we value exploration
    exploration_bonus = sigma ** beta
    
    # Combine: use a weighted combination
    # The weight on EI vs exploration can also be adaptive
    # When f_best is large, we want more exploration
    # When f_best is small, we want more exploitation
    
    # Normalize ei and exploration_bonus to comparable scales
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    exp_max = np.max(exploration_bonus) if np.max(exploration_bonus) > 0 else 1.0
    
    ei_norm = ei / ei_max
    exp_norm = exploration_bonus / exp_max
    
    # Weight: more exploration early, less later
    w_explore = 0.3 + 0.4 * min(1.0, max(0.0, f_best_val / 10.0))
    w_exploit = 1.0 - w_explore
    
    combined = w_exploit * ei_norm + w_explore * exp_norm
    
    # Divide by cost for cost-awareness
    scores = combined / np.maximum(cost, 1e-10)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
