"""Acquisition function for Bayesian optimization of Ackley 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function combining LCB and EI for Ackley 2D optimization.
    
    Uses a weighted combination of Lower Confidence Bound (exploration) and
    Expected Improvement (exploitation), with an adaptive weight based on the
    relationship between uncertainty and current best.
    
    Parameters:
    -----------
    mu : array_like, shape (n,)
        Mean of the GP posterior at candidate points.
    sigma : array_like, shape (n,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores for each candidate. Higher means evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-12)
    
    # Ensure f_best is finite
    if not np.isfinite(f_best):
        f_best = 0.0
    
    # --- Component 1: Lower Confidence Bound (LCB) ---
    # LCB = mu - beta * sigma, where beta controls exploration
    # For minimization, we want low LCB values, so higher LCB score = better
    # We use a beta that adapts based on how close f_best is to the known minimum (0)
    # When f_best is far from 0, explore more; when close, exploit more
    
    # Normalize f_best to [0, 1] range for adaptive beta
    # Ackley 2D max is around 20, min is 0
    f_best_norm = np.clip(f_best / 20.0, 0.0, 1.0)
    
    # Beta: higher when f_best is far from optimum (more exploration)
    # Range: ~1.5 to ~3.5
    beta = 1.5 + 2.0 * f_best_norm
    
    lcb = mu - beta * sigma
    
    # --- Component 2: Expected Improvement (EI) ---
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # --- Component 3: Diversity/uncertainty bonus ---
    # Reward candidates in high-uncertainty regions to encourage exploration
    # This is a mild bonus scaled by sigma
    diversity_bonus = 0.1 * sigma
    
    # --- Combine components ---
    # Weight LCB more when exploration is needed (f_best far from 0)
    # Weight EI more when exploitation is beneficial (f_best close to 0)
    
    # Weight for LCB vs EI:
    # w_lcb increases with f_best_norm (more exploration early)
    # w_ei = 1 - w_lcb
    w_lcb = 0.3 + 0.4 * f_best_norm  # Range: 0.3 to 0.7
    w_ei = 1.0 - w_lcb
    
    # Normalize LCB and EI to comparable scales
    # LCB can be negative (good), EI is non-negative
    # We want higher scores = better, so for LCB we negate (since lower LCB is better)
    # Actually, let's think about this differently:
    # For minimization:
    #   - LCB: lower is better → score = -lcb (higher = better)
    #   - EI: higher is better → score = ei
    #   - diversity: higher is better → score = diversity_bonus
    
    # But we need to combine them meaningfully. Let's use a different approach:
    # Create a unified score where higher is better.
    
    # For LCB: convert to a "desirability" score
    # lcb_desirability = -lcb (since lower lcb is better for minimization)
    # But this can be very large positive when mu is very negative
    # Let's clip and scale
    
    # Alternative: use a softmax-like combination
    # Actually, let's just use a weighted sum with proper scaling
    
    # Scale LCB to be comparable to EI
    # EI is typically in the range [0, f_best]
    # LCB can be in the range [-inf, mu_max]
    
    # Let's use a simpler approach: 
    # score = w_ei * ei + w_lcb * (-lcb) + diversity_bonus
    # But -lcb can be problematic if lcb is very negative
    
    # Better: use a bounded transformation
    # For LCB, we want to reward low values. Use:
    # lcb_score = f_best - lcb  (this is always >= 0 when lcb <= f_best)
    # But lcb can be > f_best, in which case this is negative
    # Clip to [0, inf)
    
    lcb_score = np.maximum(f_best - lcb, 0.0)
    
    # Now combine
    score = w_ei * ei + w_lcb * lcb_score + diversity_bonus
    
    # Ensure finite and non-negative
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
