import numpy as np
from scipy.stats import norm
import time

# Known global minimum of Michalewicz 2D
F_MIN = -1.8013

# Deadline guard constants
TIME_BUDGET = 320.0  # seconds, with safety margin before 330s kill
START_TIME = None

def _elapsed():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    return time.time() - START_TIME

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function blending Expected Improvement (EI) and 
    Lower Confidence Bound (LCB) for minimization.
    
    - EI focuses on exploitation (refining the current best).
    - LCB focuses on exploration (finding new promising regions).
    - An adaptive weight shifts from exploration to exploitation as f_best 
      approaches the known global minimum (-1.8013).
    
    Returns 2000 finite scores, higher meaning evaluate next.
    """
    # Deadline guard: if we've spent too much time, return a fast fallback
    if _elapsed() > TIME_BUDGET:
        # Fast fallback: simple EI
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)

    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Compute Expected Improvement (EI)
    # For minimization: EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    d_ei = (f_best - mu) / sigma_safe
    phi_d = norm.pdf(d_ei)
    Phi_d = norm.cdf(d_ei)
    
    ei = (f_best - mu) * Phi_d + sigma_safe * phi_d
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # 2. Compute Lower Confidence Bound (LCB) score for maximization
    # Standard LCB for minimization is mu - beta * sigma.
    # We want to MINIMIZE this value.
    # Since our acquisition function returns scores to MAXIMIZE, we use:
    # score_lcb = beta * sigma - mu
    # This is high when mu is low (good) and sigma is high (uncertain).
    # Use a fixed high beta for strong exploration capability
    beta = 2.5
    lcb_score = beta * sigma_safe - mu
    lcb_score = np.where(np.isfinite(lcb_score), lcb_score, -np.inf)
    
    # 3. Normalize both scores to [0, 1] to make them comparable
    # Handle case where max == min (constant score)
    def normalize_scores(s):
        s_min = np.min(s)
        s_max = np.max(s)
        if s_max - s_min < 1e-12:
            return np.ones_like(s)
        return (s - s_min) / (s_max - s_min)
    
    ei_norm = normalize_scores(ei)
    lcb_norm = normalize_scores(lcb_score)
    
    # 4. Adaptive Weighting
    # f_min_known for Michalewicz 2D is -1.8013
    f_min_known = F_MIN
    
    # Calculate progress: distance from current best to known minimum
    # d is large when we are far (early), small when close (late)
    d = f_best - f_min_known
    
    # If d is negative (we found something better than known min? unlikely but possible with noise), treat as 0
    if d < 0:
        d = 0.0
        
    # Map d to weight alpha for EI (exploitation)
    # We want alpha close to 1.0 when d is small (exploit)
    # We want alpha close to 0.2 when d is large (explore)
    # Scale: assume d=1.5 is "far enough" to be fully exploratory
    # alpha = 1.0 - 0.8 * min(1.0, d / 1.5)
    # If d=0, alpha=1.0
    # If d=1.5, alpha=0.2
    # If d>1.5, alpha=0.2
    
    alpha = 1.0 - 0.8 * min(1.0, d / 1.5)
    
    # Ensure alpha is within [0.2, 1.0]
    alpha = np.clip(alpha, 0.2, 1.0)
    
    # 5. Combine scores
    # Higher score means "evaluate next"
    # alpha * EI + (1-alpha) * LCB
    combined_score = alpha * ei_norm + (1.0 - alpha) * lcb_norm
    
    # Final safety check: ensure finite
    combined_score = np.where(np.isfinite(combined_score), combined_score, 0.0)
    
    return combined_score
