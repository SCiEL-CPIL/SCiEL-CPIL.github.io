"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # seconds


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function combining UCB and EI."""
    # Time guard: if we're close to the deadline, return a simple fallback
    elapsed = time.time() - _START_TIME
    if elapsed > 325.0:  # Safety margin before 330s limit
        # Return a simple exploration-based score
        sigma_safe = np.maximum(sigma, 1e-9)
        scores = sigma_safe / cost
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Cost-aware UCB + EI hybrid
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB component: linear exploration/exploitation trade-off
    # beta controls exploration; use a moderate value
    beta = 2.0
    ucb = mu + beta * sigma_safe
    
    # EI component: expected improvement per unit cost
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Cost is in [1, 2]. Lower cost is better.
    # For UCB (minimization), we want low mu + beta*sigma, penalized by cost
    # For EI, we want high EI / cost
    
    # Combine: use UCB for exploration, EI for exploitation
    # Scale EI to be comparable to UCB
    # UCB is in objective units, EI is also in objective units
    # But UCB is additive (mu + beta*sigma), EI is improvement-based
    
    # Let's normalize both by cost
    ucb_score = -ucb / cost  # Negate because we minimize
    ei_score = ei / cost
    
    # Weighted combination: 0.5 * UCB + 0.5 * EI
    # This balances exploration and exploitation
    scores = 0.5 * ucb_score + 0.5 * ei_score
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
