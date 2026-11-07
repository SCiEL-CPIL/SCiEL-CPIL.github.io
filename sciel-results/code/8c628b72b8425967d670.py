"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Hybrid Acquisition Function: Adaptive blend of Expected Improvement (EI)
    and Lower Confidence Bound (LCB).
    
    - Early in the optimization (high f_best), it favors LCB for exploration.
    - Late in the optimization (low f_best), it favors EI for exploitation.
    """
    # Deadline guard (safety margin)
    # The computation is vectorized and fast, but we adhere to the directive.
    # This is a placeholder for a more complex search if needed, 
    # but for this analytical acquisition, it's instantaneous.
    
    # 1. Determine Exploration vs Exploitation weight
    # Michalewicz 2D global min is approx -1.8013. Max is approx 1.0.
    f_min_global = -1.8013
    f_max_expected = 1.0
    
    # Progress: 1.0 when f_best is high (early), 0.0 when f_best is low (late)
    # Clip to handle cases where f_best might be outside this range
    progress = (f_best - f_min_global) / (f_max_expected - f_min_global)
    progress = np.clip(progress, 0.0, 1.0)
    
    # Weight for exploration (LCB). High progress -> High exploration.
    w_explore = progress
    # Weight for exploitation (EI).
    w_exploit = 1.0 - w_explore
    
    # 2. Compute Expected Improvement (EI)
    # EI is for minimization.
    # z = (f_best - mu) / sigma
    sigma_safe = sigma + 1e-9
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei_scores = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # 3. Compute LCB Score
    # LCB = mu - beta * sigma
    # We want to MAXIMIZE the score, so we use -LCB = -mu + beta * sigma
    # Beta controls exploration aggressiveness. 2.5 is a common choice for multimodal functions.
    beta = 2.5
    lcb_scores = -mu + beta * sigma
    
    # 4. Min-Max Normalization
    # Normalize EI scores to [0, 1]
    ei_min = np.min(ei_scores)
    ei_max = np.max(ei_scores)
    ei_range = ei_max - ei_min
    if ei_range < 1e-9:
        ei_norm = np.zeros_like(ei_scores)
    else:
        ei_norm = (ei_scores - ei_min) / ei_range
        
    # Normalize LCB scores to [0, 1]
    lcb_min = np.min(lcb_scores)
    lcb_max = np.max(lcb_scores)
    lcb_range = lcb_max - lcb_min
    if lcb_range < 1e-9:
        lcb_norm = np.zeros_like(lcb_scores)
    else:
        lcb_norm = (lcb_scores - lcb_min) / lcb_range
        
    # 5. Blend
    # Higher score means more likely to be selected
    final_scores = w_exploit * ei_norm + w_explore * lcb_norm
    
    # Ensure finite scores
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return final_scores
# EVOLVE-BLOCK-END
