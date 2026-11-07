"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
# 330s limit. We want to be well before.
_DEADLINE = _START_TIME + 325

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Probability of Improvement (PoI)
    with an adaptive exploration bonus, divided by cost.
    
    This approach prioritizes points that have a high probability of being
    better than the current best, while penalizing expensive points.
    It also includes an exploration term scaled by uncertainty to help escape
    local minima in multimodal functions like Rastrigin.
    
    Args:
        mu: array of shape (2000,) - GP mean
        sigma: array of shape (2000,) - GP standard deviation
        f_best: float - best observed value
        cost: array of shape (2000,) - cost of evaluating each point
        
    Returns:
        scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: Prioritize cheap points if we are running out of time
        return 1.0 / (cost + 1e-9)

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score for the probability of improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement (PoI) for minimization
    # P(f(x) < f_best) = Phi(z)
    poi = norm.cdf(z)
    
    # Adaptive exploration bonus
    # Scale exploration based on the current best value.
    # When f_best is high (early stage), we want more exploration.
    # When f_best is low (late stage), we want less exploration.
    # Using an exponential decay with a floor.
    f_best_val = float(f_best)
    
    # kappa controls the weight of the exploration term relative to PoI
    # PoI is in [0, 1]. Exploration term is normalized sigma.
    # We want exploration to be significant early on, but diminish later.
    # The scale of sigma might vary, so we normalize it.
    sigma_std = np.std(sigma)
    if sigma_std < 1e-9:
        sigma_std = 1.0
    
    # Normalized uncertainty
    exploration_term = sigma_safe / sigma_std
    
    # Adaptive beta: 
    # If f_best is large (e.g., > 10), beta is higher (more exploration).
    # If f_best is small (e.g., < 1), beta is lower (more exploitation).
    # Rastrigin minimum is 0. Typical values can be up to ~200.
    # Let's use a decay that drops off as f_best approaches 0.
    # beta = 0.5 * exp(-f_best / 10) + 0.1
    beta = 0.5 * np.exp(-f_best_val / 10.0) + 0.1
    
    # Combine PoI and Exploration
    # We want the score to be dominated by PoI but boosted by uncertainty
    numerator = poi + beta * exploration_term
    
    # Cost-aware score
    # We divide by cost to get "improvement per unit cost"
    # Using a soft denominator to avoid extreme values for very low cost (though cost >= 1)
    scores = numerator / (cost + 1e-9)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
