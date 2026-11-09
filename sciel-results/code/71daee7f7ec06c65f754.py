"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


# Deadline guard: keep explicit wall-clock tracking
_START_TIME = time.time()
_TIME_LIMIT = 300.0  # evaluator kills at 330s, leave margin
_TIME_MARGIN = 10.0  # safety margin seconds


def _time_left():
    return _TIME_LIMIT - _TIME_MARGIN - (time.time() - _START_TIME)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Branin 2D.
    
    Structural Change: Adaptive Lower Confidence Bound (LCB) blended with Expected Improvement (EI).
    
    The standard EI/UCB blend uses a fixed exploration weight. This approach adapts the 
    exploration coefficient (kappa) dynamically based on the ratio of posterior uncertainty 
    (sigma) to the spread of the GP mean (mu). 
    - High uncertainty relative to mean spread -> Higher kappa (more exploration).
    - Low uncertainty relative to mean spread -> Lower kappa (more exploitation).
    
    This helps escape local minima in Branin 2D during early/high-uncertainty phases while 
    focusing on refinement in later/low-uncertainty phases.
    
    Parameters:
    -----------
    mu : ndarray, shape (n,)
        GP mean at candidate points (objective units)
    sigma : ndarray, shape (n,)
        GP standard deviation at candidate points (objective units)
    f_best : float
        Best observed value so far
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores (higher = more promising)
    """
    # Check time budget - if we're running out, return a simple fallback
    if _time_left() < 1.0:
        # Fallback: pure EI
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 2. Compute Adaptive Lower Confidence Bound (LCB)
    # LCB score = -mu + kappa * sigma (higher is better)
    
    # Estimate adaptive kappa based on uncertainty-to-mean-spread ratio
    mu_std = np.std(mu)
    sigma_median = np.median(sigma_safe)
    
    # If mu_std is very small, the model is confident in the means.
    # If sigma is large relative to mu_std, we should explore more.
    # Ratio = sigma / mu_std. 
    # We want kappa to increase with this ratio, but bounded.
    if mu_std < 1e-6:
        # If mean is flat, rely on sigma magnitude
        # Scale sigma to a reasonable kappa range
        kappa = 1.0 + np.clip(sigma_median * 10, 0, 5.0)
    else:
        ratio = sigma_median / mu_std
        # Map ratio to kappa. 
        # ratio ~ 0.1 -> kappa ~ 1.0 (exploit)
        # ratio ~ 1.0 -> kappa ~ 2.0 (balanced)
        # ratio ~ 5.0 -> kappa ~ 5.0 (explore)
        # Linear mapping with a floor of 1.0 and ceiling of 5.0
        kappa = 1.0 + np.clip(ratio * 0.8, 0.0, 4.0)
        
    lcb = -mu + kappa * sigma_safe
    
    # 3. Normalize EI and LCB to [0, 1] for blending
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    lcb_max = np.max(lcb)
    lcb_min = np.min(lcb)
    if lcb_max - lcb_min > 1e-12:
        lcb_norm = (lcb - lcb_min) / (lcb_max - lcb_min)
    else:
        lcb_norm = np.zeros_like(lcb)
    
    # 4. Blend EI and LCB
    # Weight 0.6 on EI (exploitation) and 0.4 on LCB (adaptive exploration)
    # The adaptive nature of LCB handles the balance dynamically.
    score = 0.6 * ei_norm + 0.4 * lcb_norm
    
    # Scale back to a reasonable magnitude (EI scale) to avoid numerical issues in argmax
    # though argmax is invariant to monotonic scaling, keeping it consistent helps debugging
    if ei_max > 1e-12:
        score = score * ei_max
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
