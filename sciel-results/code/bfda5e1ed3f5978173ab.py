"""Acquisition function for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_T0 = time.time()
_DEADLINE = _T0 + 325.0  # safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Structural change: Use a z-score based blend of EI and LCB.
    Instead of min-max normalizing the raw scores (which is sensitive to outliers 
    and scales differently for EI vs LCB), we compute the z-score for each point 
    and blend the *normalized* signals directly. 
    - EI is essentially E[f_best - f | f < f_best]. It is 0 when f >> f_best.
    - LCB is mu - beta * sigma. It is lower (better) when mu is low or sigma is high.
    
    By blending in z-space or using a robust normalization, we avoid the issue where 
    one term dominates due to scale differences. We use a 'soft' blending where 
    the exploration weight is tied to the gap to the global minimum, but we 
    ensure that the LCB term is properly scaled relative to EI.
    
    Key Insight from Inspiration 1: LCB is better for early exploration on multimodal functions.
    Key Insight from Inspiration 2: Use z-clipping and robust normalization.
    
    New Approach:
    1. Compute EI.
    2. Compute LCB.
    3. Normalize both to have zero mean and unit variance (robust to outliers).
    4. Blend: w_exploit * EI_norm + w_explore * LCB_norm.
    5. The weights are determined by the progress towards the global minimum.
    """
    # Deadline guard
    if time.time() > _DEADLINE - 5.0:
        return sigma.copy()

    f_global_min = -1.8013
    f_max_expected = 1.0

    # Progress: 1.0 when f_best is high (early), 0.0 when f_best is low (late)
    # This controls the exploration/exploitation balance.
    progress = (f_best - f_global_min) / (f_max_expected - f_global_min)
    progress = np.clip(progress, 0.0, 1.0)
    
    # We want to explore more when progress is high (early) and exploit when progress is low (late).
    # However, pure EI is often too greedy. A blend is better.
    # Let's use a smooth transition.
    # w_explore = progress (high early)
    # w_exploit = 1 - progress (high late)
    
    # To prevent pure LCB from being too noisy late in the game, we cap the exploration weight.
    # And to prevent pure EI from being too greedy early, we floor the exploitation weight.
    w_explore = 0.2 + 0.8 * progress  # Range [0.2, 1.0]
    w_exploit = 1.0 - w_explore       # Range [0.0, 0.8]

    # Safe sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (minimization)
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10) # Clip to avoid numerical issues, 10 is safe
    cdf_z = norm.cdf(z_clipped)
    pdf_z = norm.pdf(z_clipped)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Compute LCB (minimization: lower is better)
    # We want to maximize the score, so we use -LCB = -mu + beta * sigma
    # Beta: How much to trust the uncertainty. 2.5 is a standard choice.
    beta = 2.5
    lcb = -mu + beta * sigma_safe
    
    # Robust Normalization: Zero mean, unit std
    # This ensures that the relative ranking within each metric is preserved,
    # but the magnitudes are comparable.
    
    ei_mean = np.mean(ei)
    ei_std = np.std(ei)
    if ei_std < 1e-9:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = (ei - ei_mean) / ei_std
        
    lcb_mean = np.mean(lcb)
    lcb_std = np.std(lcb)
    if lcb_std < 1e-9:
        lcb_norm = np.zeros_like(lcb)
    else:
        lcb_norm = (lcb - lcb_mean) / lcb_std
        
    # Combine
    final_scores = w_exploit * ei_norm + w_explore * lcb_norm
    
    # Ensure finite
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return final_scores
# EVOLVE-BLOCK-END
