import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function for Bayesian optimization.
    
    Uses a dynamic exploration coefficient that adapts based on the uncertainty landscape,
    then normalizes by evaluation cost to prioritize high-value, low-cost candidates.
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clamp sigma to avoid division by zero and numerical instability
    sigma_safe = np.clip(sigma, 1e-10, None)
    
    # Clamp cost to avoid division by zero
    cost_safe = np.clip(cost, 1e-10, None)
    
    # Compute dynamic exploration coefficient
    # Use a heuristic based on the ratio of max uncertainty to typical uncertainty
    # This helps balance exploration vs exploitation adaptively
    sigma_mean = np.mean(sigma_safe)
    sigma_max = np.max(sigma_safe)
    
    # Dynamic beta: scale with the spread of uncertainty
    # When uncertainty is high relative to mean, explore more
    if sigma_mean > 1e-10:
        uncertainty_ratio = sigma_max / (sigma_mean + 1e-10)
    else:
        uncertainty_ratio = 1.0
    
    # Base beta with adaptive scaling
    # Start with a moderate exploration coefficient and scale based on uncertainty spread
    base_beta = 2.0
    # Increase exploration when uncertainty is more varied (higher max/mean ratio)
    adaptive_factor = min(uncertainty_ratio, 3.0) / 1.0  # Cap at 3x
    beta = base_beta * adaptive_factor
    
    # Also consider the overall uncertainty level relative to f_best
    # If f_best is close to 0 (good), we can be more conservative
    if f_best > 0:
        # Scale beta down slightly when we're doing well
        improvement_scale = 1.0 / (1.0 + f_best)
        beta *= improvement_scale
    
    # Compute LCB: mu - beta * sigma
    # Lower is better for the objective, so we want to minimize this
    lcb = mu - beta * sigma_safe
    
    # The acquisition score should be higher for points we want to evaluate
    # Since lower LCB means better predicted value, we negate it
    scores = -lcb / cost_safe
    
    # Add a small exploration bonus to encourage evaluating high-uncertainty, low-cost points
    # This helps the GP model converge faster
    exploration_bonus = 0.05 * sigma_safe / cost_safe
    scores += exploration_bonus
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard: if this took too long, return a simpler heuristic
    elapsed = time.time() - start_time
    if elapsed > 250.0:
        # Fallback: simple LCB per cost
        simple_beta = 2.0
        simple_lcb = mu - simple_beta * sigma_safe
        simple_scores = -simple_lcb / cost_safe
        simple_scores = np.where(np.isfinite(simple_scores), simple_scores, 0.0)
        return simple_scores
    
    return scores
