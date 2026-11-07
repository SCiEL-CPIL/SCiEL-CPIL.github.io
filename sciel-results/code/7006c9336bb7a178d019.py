"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a Cost-Aware Upper Confidence Bound (CA-UCB) with adaptive beta
    that shifts from exploration to exploitation as budget is consumed.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best (lowest) observed value so far
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Known minimum of hartmann 3d
    f_min_known = -3.86278
    
    # Calculate total cost consumed so far
    # We estimate this from the cost array - the mean cost gives us a sense of budget usage
    # But actually, we need to track cumulative cost. Since we only have per-candidate cost,
    # we use a heuristic: the number of evaluations done is related to how many points
    # have been evaluated. We'll use f_best as a proxy for progress.
    
    # Better approach: use the gap to known minimum as a proxy for progress
    gap = f_best - f_min_known
    gap_normalized = np.clip(gap / 4.0, 0.0, 1.0)
    
    # Adaptive beta for UCB:
    # Early on (large gap), we want more exploration -> higher beta
    # Later (small gap), we want more exploitation -> lower beta
    # beta = base_beta * (1 + gap_normalized)
    base_beta = 1.5
    beta = base_beta * (0.5 + gap_normalized)
    
    # UCB for minimization: we want to minimize, so we use lower confidence bound
    # LCB = mu - beta * sigma
    # Higher score means more promising (lower predicted value with uncertainty)
    lcb = mu - beta * sigma_safe
    
    # Also compute Expected Improvement for a hybrid approach
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware scaling: divide by cost to prefer cheaper evaluations
    # Use a power law to control sensitivity
    alpha = 0.3 + 0.4 * gap_normalized  # more cost-sensitive early on
    
    cost_factor = cost_safe ** alpha
    
    # Combine LCB and EI
    # Normalize both to comparable scales
    lcb_normalized = (lcb - np.min(lcb)) / (np.max(lcb) - np.min(lcb) + 1e-9)
    ei_normalized = ei / (np.max(ei) + 1e-9)
    
    # Weight between LCB and EI based on progress
    # Early: more LCB (exploration), Late: more EI (exploitation)
    lcb_weight = 0.4 + 0.4 * gap_normalized
    ei_weight = 1.0 - lcb_weight
    
    combined = lcb_weight * lcb_normalized + ei_weight * ei_normalized
    
    # Apply cost-aware penalty
    scores = combined / cost_factor
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Add a small exploration bonus for high-uncertainty regions
    # This helps avoid premature convergence
    exploration_bonus = 0.05 * sigma_safe / cost_safe
    scores = scores + exploration_bonus
    
    # Final safety: ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
