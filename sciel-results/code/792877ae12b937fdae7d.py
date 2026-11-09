# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Uses a hybrid approach: primarily LCB for exploration with a small
    EI component for exploitation, plus an adaptive uncertainty bonus
    to balance exploration/exploitation over the course of optimization.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : np.ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far.
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores; higher values are preferred.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute Lower Confidence Bound (LCB) with adaptive beta
    # Beta starts high for exploration, decreases as we get better estimates
    # We estimate progress from how close f_best is to the known minimum
    min_val = -1.8013
    # Progress: 0 at start (f_best is far from min), 1 when we've found min
    progress = np.clip((f_best - min_val) / (f_best - min_val + 1e-9), 0, 1)
    # Actually, let's use a simpler heuristic: start with high exploration
    # and reduce it. We don't know the iteration count, so use f_best magnitude.
    # For Michalewicz, typical starting values are around -0.5 to 0.
    # As we improve toward -1.8, we can reduce exploration.
    
    # Adaptive beta: start at 3.0 (heavy exploration), decay to 1.5
    # Use f_best to estimate progress: if f_best is close to min, we're doing well
    # Scale: when f_best = 0, beta = 3.0; when f_best = min_val, beta = 1.5
    span = 0 - min_val  # ~1.8013
    if span > 1e-9:
        t = np.clip((f_best - min_val) / span, 0, 1)  # 1 at start, 0 at min
        beta = 1.5 + 1.5 * t  # 3.0 at start, 1.5 at min
    else:
        beta = 1.5
    
    lcb = mu - beta * sigma
    
    # Hybrid: weight EI and LCB
    # Early on, favor LCB (exploration); later, favor EI (exploitation)
    # Use the same t for weighting
    if span > 1e-9:
        t = np.clip((f_best - min_val) / span, 0, 1)
        w_ei = 0.3 * (1 - t) + 0.1 * t  # 0.3 early, 0.1 late
        w_lcb = 1 - w_ei  # 0.7 early, 0.9 late
    else:
        w_ei = 0.1
        w_lcb = 0.9
    
    # Normalize EI and LCB to similar scales for combination
    # EI is typically smaller in magnitude; scale it up
    ei_scale = np.mean(np.abs(ei)) + 1e-9
    lcb_scale = np.mean(np.abs(lcb)) + 1e-9
    
    scores = w_lcb * (lcb / lcb_scale) + w_ei * (ei / ei_scale)
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
