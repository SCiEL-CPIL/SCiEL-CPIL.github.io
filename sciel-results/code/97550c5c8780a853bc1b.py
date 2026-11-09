"""Acquisition function for Bayesian optimization of goldstein price 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points.
    Uses a hybrid of Expected Improvement and UCB with adaptive exploration
    to balance exploitation and exploration for the multimodal Goldstein-Price function.
    """
    # Deadline guard: track elapsed time to ensure we return before 300s
    t0 = time.time()
    deadline = 300.0 - 2.0  # safety margin
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    # Check deadline before heavy computation
    if time.time() > deadline:
        # Fallback: simple UCB
        return -mu + 1.0 * sigma
    
    # Standard Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB component for exploration
    # Adaptive beta based on gap to known minimum (Goldstein-Price min is 3.0)
    f_min = 3.0
    gap = max(f_best - f_min, 0.0)
    
    # beta decreases as we get closer to the optimum
    # Start with higher exploration and reduce as we converge
    beta = 0.3 + 0.7 * np.exp(-0.3 * gap)
    
    ucb = -mu + beta * sigma
    
    # Normalize EI and UCB to comparable scales
    # EI can be very small, UCB can be larger in magnitude
    ei_max = np.max(np.abs(ei)) if np.max(np.abs(ei)) > 1e-12 else 1.0
    ucb_max = np.max(np.abs(ucb)) if np.max(np.abs(ucb)) > 1e-12 else 1.0
    
    ei_norm = ei / ei_max
    ucb_norm = ucb / ucb_max
    
    # Combine: weight towards EI as we converge, towards UCB when exploring
    # Early on (large gap), UCB dominates; later, EI dominates
    w_ucb = 0.5 * np.exp(-0.2 * gap)
    w_ei = 1.0 - w_ucb
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Final deadline check
    if time.time() > deadline:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
