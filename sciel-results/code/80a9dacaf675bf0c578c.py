"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Combines Expected Improvement (EI) with a UCB term, weighted adaptively
    based on the gap between current best and the best predicted mean.
    
    - EI: Exploits regions where improvement over f_best is likely.
    - UCB (score = -mu + k*sigma): Explores regions of high uncertainty.
    
    The blend weight shifts from exploration to exploitation as the gap
    between f_best and min(mu) shrinks relative to the uncertainty scale.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)  where z = (f_best - mu) / sigma
    delta = f_best - mu
    z = delta / sigma_safe
    z_clipped = np.clip(z, -30, 30)
    ei = delta * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # UCB score for minimization: we want to maximize this
    # Optimistic value: mu - k*sigma, so score = -(mu - k*sigma) = -mu + k*sigma
    k = 1.0
    ucb_score = -mu + k * sigma_safe
    
    # Determine progress: 0 = early (large gap), 1 = late (small gap)
    min_mu = np.min(mu)
    gap = f_best - min_mu
    max_sigma = np.max(sigma_safe)
    
    if max_sigma > 1e-12:
        # Normalize gap by max sigma; clip to [0, 1]
        # When gap is large relative to sigma, progress is low (explore)
        # When gap is small relative to sigma, progress is high (exploit)
        raw_progress = gap / (max_sigma + 1e-12)
        progress = np.clip(raw_progress, 0.0, 1.0)
    else:
        progress = 1.0
    
    # Smooth transition: use a logistic-like function
    # exploration_weight: high when progress is low, low when progress is high
    # exploitation_weight: complementary
    # Use a sigmoid centered at 0.5
    t = 5.0 * (progress - 0.5)
    exploration_weight = 1.0 / (1.0 + np.exp(t))
    exploitation_weight = 1.0 - exploration_weight
    
    # Normalize each score to [0, 1] for meaningful blending
    ei_min, ei_max = np.min(ei), np.max(ei)
    if ei_max - ei_min > 1e-12:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.full_like(ei, 0.5)
    
    ucb_min, ucb_max = np.min(ucb_score), np.max(ucb_score)
    if ucb_max - ucb_min > 1e-12:
        ucb_norm = (ucb_score - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.full_like(ucb_score, 0.5)
    
    # Blend
    score = exploitation_weight * ei_norm + exploration_weight * ucb_norm
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
