"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # UCB with adaptive exploration: balances exploitation (low mu) with exploration (high sigma)
    # For minimization, we want low mu and high sigma.
    # Normalize sigma to [0, 1] range to make beta scale-invariant
    sigma_max = np.max(sigma)
    if sigma_max < 1e-12:
        sigma_norm = sigma
    else:
        sigma_norm = sigma / sigma_max
    
    # Adaptive beta: start high for exploration, decay as we find better solutions
    # f_best starts high (bad) and decreases (good), so (1 - f_best) increases as we improve
    # Use a tanh-based decay to keep beta in [0.5, 2.0]
    beta = 0.5 + 1.5 * np.tanh(1.0 - f_best)
    
    # UCB for minimization: score = -mu + beta * sigma_norm
    # Higher score means more promising
    scores = -mu + beta * sigma_norm
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
