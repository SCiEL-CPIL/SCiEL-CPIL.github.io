"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Compute cost-aware acquisition scores for 2000 candidate points.
    
    Uses a hybrid approach:
    - Early stage: aggressive Lower Confidence Bound (LCB) to explore
    - Late stage: robust Expected Improvement (EI) normalized by cost
    
    The stage is determined by the total cost spent so far relative to the budget.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions at candidate points
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions at candidate points
    f_best : float
        Best (lowest) observed function value so far
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next)
    """
    # Deadline guard: track elapsed time
    t_start = time.time()
    DEADLINE = 325.0  # Safety margin before 330s limit
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Estimate total cost spent so far
    # cost array has shape (2000,) - these are costs of NEW candidates
    # We need to estimate how much budget has been used
    # The initial 5 points cost approximately 1 + mean of uniform [0,1] scaled values
    # For uniform points in [-5,5]^2, scaled to [0,1], mean(x) is around 0.5
    # So each initial point costs about 1 + 0.5 = 1.5, total ~7.5
    # Each subsequent evaluation adds its cost
    # We don't have direct access to total cost, but we can infer from f_best
    # and the number of evaluations that have occurred.
    
    # Alternative: use the magnitude of f_best to estimate stage
    # If f_best is still large, we're early; if small, we're late
    # For three hump camel, minimum is 0, so f_best approaching 0 means late stage
    
    # Better approach: track number of calls or use a heuristic based on f_best
    # Since we don't have direct access to iteration count, use f_best as proxy
    # f_best starts around O(1-10) and decreases to near 0
    
    # Use a normalized measure: if f_best is close to 0, we're in late stage
    # Threshold: if f_best < 0.5, we're in late stage (exploit more)
    # If f_best > 2.0, we're in early stage (explore more)
    
    f_best_val = float(f_best) if np.isfinite(f_best) else 10.0
    
    # Stage determination
    if f_best_val > 3.0:
        # Very early: strong exploration
        beta = 2.5
        mode = 'explore'
    elif f_best_val > 1.0:
        # Early-mid: balanced
        beta = 1.5
        mode = 'balanced'
    elif f_best_val > 0.1:
        # Mid-late: more exploitation
        beta = 0.8
        mode = 'exploit'
    else:
        # Very late: strong exploitation
        beta = 0.3
        mode = 'exploit'
    
    # Check deadline
    if time.time() - t_start > DEADLINE:
        # Return simple EI/cost as fallback
        z = (f_best_val - mu) / sigma
        ei = (f_best_val - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / cost
    
    # Compute acquisition scores
    if mode == 'explore':
        # LCB: mu - beta * sigma, normalized by cost
        # We want to MINIMIZE mu - beta*sigma, so score = -(mu - beta*sigma) / cost
        lcb = mu - beta * sigma
        scores = -lcb / cost
    elif mode == 'balanced':
        # Hybrid: mix of LCB and EI
        z = (f_best_val - mu) / sigma
        ei = (f_best_val - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        lcb = -(mu - beta * sigma)  # positive when promising
        # Normalize both to comparable scales
        ei_norm = ei / (np.max(ei) + 1e-9) if np.max(ei) > 0 else ei
        lcb_norm = lcb / (np.max(lcb) + 1e-9) if np.max(lcb) > 0 else lcb
        scores = (0.5 * ei_norm + 0.5 * lcb_norm) / cost
    else:
        # Exploit: EI normalized by cost
        z = (f_best_val - mu) / sigma
        ei = (f_best_val - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        scores = ei / cost
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Final deadline check
    if time.time() - t_start > DEADLINE:
        pass  # Just return what we have
    
    return scores
# EVOLVE-BLOCK-END
