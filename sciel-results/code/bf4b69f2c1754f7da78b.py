"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Adaptive acquisition for Bayesian optimization of Ackley 2D.
    
    Combines Expected Improvement (EI) with a UCB-like exploration term.
    The blend shifts from exploration-heavy early on to exploitation-heavy
    as f_best approaches 0, navigating Ackley's multimodal landscape.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Posterior mean predictions at candidate points
    sigma : array of shape (2000,)
        Posterior standard deviation predictions at candidate points
    f_best : float
        Best observed value so far (lowest)
    
    Returns
    -------
    scores : array of shape (2000,)
        Acquisition scores, higher values indicate more promising candidates
    """
    start_time = time.time()
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    # Clip f_best to avoid issues
    f_best_clipped = max(abs(f_best), 1e-9)
    
    # Compute Expected Improvement
    z = (f_best_clipped - mu) / sigma
    ei = (f_best_clipped - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Compute UCB-like exploration term
    # Use a beta that transitions from exploration-heavy to exploitation-heavy
    # As f_best -> 0, we want pure exploitation (EI dominates)
    # As f_best is large, we want more exploration (UCB contributes more)
    
    # Smooth transition: beta goes from ~1 (high exploration) to ~0 (low exploration)
    # Use exponential decay: beta = exp(-f_best / scale)
    # When f_best is small, beta ~ 1 (high exploration)
    # When f_best is large, beta ~ 0 (low exploration)
    scale = 2.0
    beta = np.exp(-f_best_clipped / scale)
    
    # UCB term: sigma scaled by a constant
    # Use a moderate coefficient to balance exploration
    ucb_term = 1.0 * sigma
    
    # Combine EI and UCB with adaptive weighting
    # weights: EI gets (1-beta), UCB gets beta
    # But we want EI to always have a strong presence
    # So: score = ei + beta * ucb_term
    # This ensures EI is always the base, with exploration added on top
    
    scores = ei + beta * ucb_term
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety check: if computation took too long, return early
    elapsed = time.time() - start_time
    if elapsed > 250:  # Safety margin before 300s limit
        # Return best effort - just the EI score
        scores = ei
        scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
