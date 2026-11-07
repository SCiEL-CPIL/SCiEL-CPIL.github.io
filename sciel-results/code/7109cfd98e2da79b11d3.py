import numpy as np
from scipy.stats import norm
import time

# ---------- Global timer / safety ----------
_START_TIME = time.time()
_TIME_LIMIT = 325.0  # Safety margin below 330s wall-clock

# ---------- Acquisition Parameters ----------
# These are chosen based on standard cost-aware BO heuristics for this problem scale.
# No simulation-based fitting to ensure speed and reliability.
_BETA = 2.0       # Increased exploration to find better minima
_W_EI = 0.6       # Slightly reduce EI weight
_W_UCL = 0.4      # Increase UCL weight for more exploration
_COST_POW = 1.0   # Linear cost penalty (less aggressive than 1.2)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    Returns scores of shape (2000,), higher = evaluate next.
    """
    # Explicit deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback: return EI / Cost
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        score = ei / cost_safe
        score = np.where(np.isfinite(score), score, -1e9)
        return score

    beta = _BETA
    w_ei = _W_EI
    w_ucl = _W_UCL
    cost_pow = _COST_POW

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Upper Confidence Bound (UCL) potential
    # We want to minimize, so a low UCL is good.
    # UCL = mu - beta * sigma
    # Potential = f_best - UCL = f_best - (mu - beta * sigma) = (f_best - mu) + beta * sigma
    # Add a small bonus for high uncertainty to encourage exploration in unexplored regions
    ucl_potential = (f_best - mu) + beta * sigma_safe
    # Normalize by typical scale to make UCL contribution comparable to EI
    # Use f_best as a rough scale reference (assuming f_best is O(1))
    ucl_potential = ucl_potential * (1.0 / max(abs(f_best), 0.1))
    
    # Blend EI and UCL potential
    blended_score = w_ei * ei + w_ucl * ucl_potential
    
    # Apply cost penalty
    # Higher cost -> lower score
    # Use a softer penalty that doesn't completely dominate the acquisition signal
    cost_penalty = cost_safe ** cost_pow
    score = blended_score / cost_penalty
    
    # Add a small constant to avoid zero scores when blended_score is near zero
    # This helps with numerical stability and ensures all points have some score
    score = score + 1e-6
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -1e9)
    
    return score
