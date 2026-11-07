"""Acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) with a
    normalized exploration term (Normalized EI / sigma).
    
    mu: array of shape (2000,) - posterior mean
    sigma: array of shape (2000,) - posterior std
    f_best: float - best observed value so far
    
    Returns: array of shape (2000,) - acquisition scores (higher = more promising)
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalized EI: divide by sigma to get a scale-free exploration measure
    # This balances exploitation (high EI) with exploration (high uncertainty)
    # The normalized form adapts to local uncertainty scale
    nee = ei / sigma_safe  # This is equivalent to z * cdf(z) + pdf(z)
    
    # Combine: use a weighted sum where the weight adapts based on how much
    # improvement is expected. When EI is small relative to uncertainty,
    # exploration dominates; when EI is large, exploitation dominates.
    # 
    # Use a soft blend: score = ei + beta * nee * sigma_safe
    # But since nee * sigma_safe = ei, that's just 2*ei.
    # 
    # Better: use the standard EI but add a small UCB-like exploration bonus
    # scaled by the current uncertainty level.
    #
    # Actually, the measured finding shows NEI (EI/sigma) improved from 0.678 to 1.657.
    # So let's use a combination: primarily EI, with a normalization-based exploration.
    
    # Strategy: Use EI as the primary signal, but add an exploration bonus
    # that is proportional to sigma when EI is small (early in optimization).
    # As f_best improves, EI becomes more reliable and exploration should decrease.
    
    # Compute a baseline EI score
    ei_score = ei.copy()
    
    # Exploration bonus: use normalized EI (which is dimensionless)
    # Scale it so it's comparable to EI
    # The typical scale of EI is on the order of the objective's std
    # Let's estimate the scale from the data
    ei_scale = np.median(np.abs(ei_score[ei_score > 0])) if np.any(ei_score > 0) else 1.0
    if ei_scale < 1e-10:
        ei_scale = 1.0
    
    # Blend: start with heavy exploration, reduce as we find good points
    # Use a sigmoid-like transition based on how many points have positive EI
    n_positive = np.sum(ei_score > 0)
    frac_positive = n_positive / len(ei_score)
    
    # When few points have positive EI (early), explore more
    # When many points have positive EI (later), exploit more
    # beta transitions from ~1.0 to ~0.1
    beta = 0.1 + 0.9 * (1.0 - frac_positive)
    
    # Final score: EI + beta * (normalized EI scaled back to EI units)
    # normalized EI * ei_scale gives us an exploration term in comparable units
    exploration_term = beta * nee * ei_scale
    
    scores = ei_score + exploration_term
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
