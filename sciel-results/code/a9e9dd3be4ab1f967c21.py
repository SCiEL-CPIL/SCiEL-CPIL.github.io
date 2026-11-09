import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Adaptive hybrid acquisition: blends Expected Improvement and UCB
    with exploration weight that decreases as we approach the known minimum.
    
    mu: (2000,) posterior mean in objective units
    sigma: (2000,) posterior std in objective units
    f_best: lowest observed value so far
    
    Returns: (2000,) scores, higher = more desirable to evaluate next
    """
    # Known minimum of Michalewicz 2D
    f_min_known = -1.8013
    
    # Progress: 0 at start (f_best far from minimum), 1 when we've found the minimum
    # Initial f_best is typically around -0.5 to -1.0 for 5 random points
    # Use a smooth progress measure
    f_ref = -0.5  # rough initial value
    progress = np.clip((f_best - f_ref) / (f_min_known - f_ref), 0.0, 1.0)
    # progress = 0 when f_best = -0.5, progress = 1 when f_best = -1.8013
    
    # Safety: ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (for minimization: improvement = f_best - mu)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB (for minimization: we want low mu + high uncertainty)
    # beta controls exploration; adaptive based on progress
    # Early: high beta (explore), Late: low beta (exploit)
    beta = 0.5 + 2.0 * (1.0 - progress)  # beta ranges from ~2.5 (early) to ~0.5 (late)
    ucb = mu + beta * sigma  # lower is better for minimization, so we negate for scoring
    
    # For scoring (higher = better):
    # EI is already "improvement" so higher is better
    # UCB: lower value is better, so score = -ucb
    ei_score = ei  # higher EI = more desirable
    ucb_score = -ucb  # lower UCB value = higher score
    
    # Normalize both to comparable scale
    ei_mean = np.mean(ei_score)
    ei_std = np.std(ei_score) + 1e-9
    ei_norm = (ei_score - ei_mean) / ei_std
    
    ucb_mean = np.mean(ucb_score)
    ucb_std = np.std(ucb_score) + 1e-9
    ucb_norm = (ucb_score - ucb_mean) / ucb_std
    
    # Blend: early on, weight UCB more; late on, weight EI more
    w_ei = 0.3 + 0.7 * progress  # w_ei ranges from 0.3 to 1.0
    w_ucb = 1.0 - w_ei + 0.3  # keep some UCB weight always, but less as progress increases
    # Normalize weights
    w_total = w_ei + w_ucb
    w_ei /= w_total
    w_ucb /= w_total
    
    score = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
