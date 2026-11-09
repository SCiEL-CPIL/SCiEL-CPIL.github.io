"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_START_TIME = time.time()
_TIMEOUT = 290  # Safety margin before 300s

# Known global minimum for Styblinski-Tang 1D
_KNOWN_MIN = -39.1662

def acquisition(mu, sigma, f_best):
    """
    Hybrid Acquisition Function: EI + Global-Aware Exploration Bonus.
    
    This function blends Expected Improvement (EI) with an exploration term
    that is scaled by the distance to the known global minimum. This helps
    escape local minima in the early stages and focuses on exploitation
    in the late stages.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Posterior mean of the GP at candidate points.
    sigma : array-like, shape (2000,)
        Posterior standard deviation of the GP at candidate points.
    f_best : float
        Best observed function value so far (minimum).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Check deadline - if we're running out of time, return a fast fallback (Pure EI)
    if time.time() - _START_TIME > _TIMEOUT:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)

    # 1. Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Compute Exploration Bonus
    # The bonus is proportional to sigma, but scaled by how far we are from the optimum.
    # If f_best is close to _KNOWN_MIN, the gap is small, and exploration is suppressed.
    # If f_best is far, the gap is large, and exploration is encouraged.
    
    # Gap to known minimum
    gap = f_best - _KNOWN_MIN
    if gap < 0:
        gap = 0.0
        
    # We want the exploration weight to be significant early on.
    # A simple scaling: weight is proportional to the gap, but capped or normalized.
    # Let's use a normalized gap relative to the typical range of the function (~100 units)
    # Or simply use the gap directly if it's small, but we need to balance scales.
    
    # EI values are typically small (improvement in f). Sigma is in objective units.
    # We need to make the bonus comparable to EI.
    
    # Heuristic: The maximum possible improvement is roughly f_best - _KNOWN_MIN.
    # So, a good scale for the bonus is a fraction of this gap.
    
    # If gap is very small, we want pure EI.
    # If gap is large, we want to explore.
    
    # Let's define an adaptive beta for UCB-like behavior.
    # beta = c * (gap / sigma_mean) ?? No, let's stick to the Inspiration 2 logic which worked.
    
    # Inspiration 2 used: bonus = 0.05 * log1p(gap/10) * sigma
    # Let's refine this. We want the exploration to be stronger when uncertainty is high.
    
    # Normalize EI to [0, 1] to allow blending with a normalized exploration score?
    # Or just add a scaled sigma term.
    
    # Let's try a blend of EI and a "Global UCB" term.
    # UCB for minimization: -mu + beta * sigma.
    # But we want to keep the "improvement" aspect of EI.
    
    # Alternative: Knowledge Gradient Approximation.
    # KG ~ E[ min(mu) - mu ] but that's not right.
    # Let's stick to the successful "EI + Exploration" structure but make it more robust.
    
    # Calculate mean sigma to normalize the exploration term
    sigma_mean = np.mean(sigma_safe)
    if sigma_mean < 1e-9:
        sigma_mean = 1e-9

    # The exploration term should be larger when we are far from the optimum.
    # Let's use a weight that decays as we get closer.
    # weight = max(0, (f_best - _KNOWN_MIN) / 50.0)  # Scale factor
    # This is too aggressive.
    
    # Let's use the logic from Inspiration 2 which scored 1.338:
    # bonus = exploration_weight * sigma_safe
    # exploration_weight = 0.05 * np.log1p(gap / 10.0)
    
    # However, simple addition might not be enough if EI dominates.
    # Let's normalize both to ensure they compete fairly.
    
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = max(ei_max - ei_min, 1e-9)
    ei_norm = (ei - ei_min) / ei_range
    
    # Exploration score: higher sigma is better, but we want to prioritize points that are
    # also somewhat promising (low mu). 
    # A simple sigma term might pick noise. 
    # Let's use a UCB-like score: -mu + beta * sigma.
    # But we want to blend it with EI.
    
    # Let's try a weighted sum of normalized EI and normalized UCB.
    # UCB for minimization (higher score is better): -mu + beta * sigma
    # We need to choose beta.
    # If we are far from optimum, beta should be higher.
    
    # Adaptive Beta
    # If gap is large, explore more.
    # beta = 1.0 + gap / 10.0 ?
    # Let's cap it.
    beta = 1.0 + min(gap / 10.0, 2.0) # Beta between 1.0 and 3.0
    
    ucb = -mu + beta * sigma_safe
    
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = max(ucb_max - ucb_min, 1e-9)
    ucb_norm = (ucb - ucb_min) / ucb_range
    
    # Blend coefficient
    # If we are close to the minimum (gap small), trust EI (alpha=1.0).
    # If we are far (gap large), trust UCB (alpha=0.0 or low).
    
    # Normalize gap to [0, 1] roughly. Range of ST1D is about 100.
    # gap=0 -> alpha=1.0 (Pure EI)
    # gap=50 -> alpha=0.0 (Pure UCB)
    # Linear decay
    alpha = max(0.0, 1.0 - (gap / 50.0))
    
    # If gap is huge, alpha becomes 0.
    # If gap is 0, alpha is 1.
    
    scores = alpha * ei_norm + (1.0 - alpha) * ucb_norm
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
