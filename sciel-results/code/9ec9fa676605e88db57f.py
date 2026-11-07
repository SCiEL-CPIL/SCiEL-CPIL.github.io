"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Enhanced Probability of Improvement (POI) based acquisition function.
    
    Uses a blend of POI and Expected Improvement to balance exploration
    and exploitation more effectively.
    
    The score combines:
    1. Probability of improvement: Phi((f_best - mu) / sigma)
    2. Expected improvement magnitude: (f_best - mu) * Phi(z) + sigma * pdf(z)
    
    We use a weighted combination that emphasizes points with both high
    probability of improvement and large expected gains.
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized distance
    z = (f_best - mu) / sigma_safe
    
    # Probability of improvement
    poi = norm.cdf(z)
    
    # Expected improvement
    ei = (f_best - mu) * poi + sigma_safe * norm.pdf(z)
    
    # Combine POI and EI with weights that favor exploitation when
    # we have high confidence, and exploration when uncertain
    # Use a geometric mean-like combination to balance both
    # Score = POI * (1 + EI / (f_best - mu + eps)) when f_best > mu
    # But simpler: weight EI by POI to get a "confident improvement" score
    
    # Alternative: Use EI directly but with a modified form
    # that emphasizes the probability component
    # score = EI + alpha * POI * sigma
    # This adds an exploration bonus proportional to uncertainty
    
    # Let's use a simpler, more robust approach:
    # score = EI, but ensure numerical stability
    # EI is already a well-known acquisition function
    
    # However, the task asks for POI-based, so let's stay with POI
    # but improve the scaling
    
    # Key insight: (f_best - mu) * POI can be problematic when f_best - mu
    # is negative (mu > f_best), giving negative scores. We should clip.
    
    # Better approach: Use max(0, f_best - mu) * POI + small exploration bonus
    gain = np.maximum(f_best - mu, 0)
    
    # Primary score: probability-weighted gain
    score = gain * poi
    
    # Add a small exploration bonus for high-uncertainty regions
    # where there's still a reasonable chance of improvement
    exploration_bonus = 0.1 * sigma_safe * poi
    
    score = score + exploration_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
