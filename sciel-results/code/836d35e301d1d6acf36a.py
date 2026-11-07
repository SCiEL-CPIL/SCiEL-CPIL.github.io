import numpy as np
from scipy.stats import norm
import time

# Global timer to track elapsed time across calls
_start_time = None

def acquisition(mu, sigma, f_best):
    """
    Variance-Weighted Expected Improvement (VW-EI) with adaptive exploration.
    
    This acquisition function combines:
    1. Standard Expected Improvement (EI) for exploitation
    2. Variance-weighted scaling to prioritize uncertain regions
    3. A small UCB-like exploration bonus
    
    The key structural change from standard EI is multiplying by normalized variance,
    which converts the objective to "expected improvement per unit of model uncertainty."
    This aggressively prioritizes candidates where the model is most uncertain,
    helping escape local minima in multimodal landscapes like Styblinski-Tang.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we're running long, return a simple score
    elapsed = time.time() - _start_time
    if elapsed > 300:
        # Fallback: simple UCB-like score
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + 1.0 * sigma_safe
    
    # Handle division by zero or very small sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0)
    
    # Compute normalized variance: sigma^2 / max(sigma^2)
    sigma_sq = sigma_safe ** 2
    sigma_sq_max = np.max(sigma_sq)
    if sigma_sq_max > 1e-12:
        var_norm = sigma_sq / sigma_sq_max
    else:
        var_norm = np.ones_like(sigma_sq)
    
    # Variance-Weighted EI: multiply EI by normalized variance
    # This prioritizes points where the model is uncertain about improvement
    vw_ei = ei * var_norm
    
    # Also compute standard EI normalized for blending
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    vw_ei_max = np.max(vw_ei)
    if vw_ei_max > 1e-12:
        vw_ei_norm = vw_ei / vw_ei_max
    else:
        vw_ei_norm = np.zeros_like(vw_ei)
    
    # UCB component for exploration (negative because we minimize)
    # Use a moderate beta for exploration
    beta = 0.3
    ucb = -mu + beta * sigma_safe
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    if ucb_max - ucb_min > 1e-12:
        ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Adaptive weighting: favor VW-EI more when EI is low (exploration phase)
    # and standard EI more when EI is high (exploitation phase)
    # Use the mean normalized EI as a proxy for exploitation strength
    mean_ei_norm = np.mean(ei_norm)
    
    # When mean EI is low, we're in exploration mode -> weight VW-EI more
    # When mean EI is high, we're in exploitation mode -> weight standard EI more
    exploration_weight = np.clip(1.0 - mean_ei_norm * 2.0, 0.3, 0.7)
    exploitation_weight = 1.0 - exploration_weight
    
    # Blend VW-EI and standard EI
    blended_ei = exploration_weight * vw_ei_norm + exploitation_weight * ei_norm
    
    # Add a small UCB component for additional exploration
    ucb_weight = 0.15
    score = (1.0 - ucb_weight) * blended_ei + ucb_weight * ucb_norm
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
